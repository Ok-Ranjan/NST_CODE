import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import uuid
import threading
import time
import traceback
from collections import OrderedDict

import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

# import existing AdaIn code
from src.encoder import VGGEncoder
from src.decoder import Decoder
from src.adain import adaptive_instance_normalization, calc_mean_std

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit
csrf = CSRFProtect(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory job store  (OrderedDict; capped at 100 entries)
# ---------------------------------------------------------------------------
_jobs: OrderedDict = OrderedDict()
_jobs_lock = threading.Lock()
MAX_JOBS = 100

_inference_queue: list = []
_queue_lock = threading.Lock()
_worker_thread = None
_worker_event = threading.Event()


def _get_or_start_worker():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()


def _worker_loop():
    while True:
        _worker_event.wait(timeout=60)
        _worker_event.clear()
        while True:
            with _queue_lock:
                if not _inference_queue:
                    break
                job_id, content_path, style_path, alpha = _inference_queue.pop(0)
            _run_inference(job_id, content_path, style_path, alpha)


def _run_inference(job_id, content_path, style_path, alpha):
    _set_job(job_id, status='running')
    content_tensor = style_tensor = stylized_image = None
    content_feats = style_feats = stylized_feats = None
    try:
        t0 = time.perf_counter()
        content_image = Image.open(content_path).convert('RGB')
        style_image = Image.open(style_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        content_tensor = transform(content_image).unsqueeze(0).to(device)
        style_tensor = transform(style_image).unsqueeze(0).to(device)

        with torch.no_grad():
            content_feats = encoder(content_tensor, test=True)
            style_feats = encoder(style_tensor, test=True)
            stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
            stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats
            stylized_image = decoder(stylized_feats)

        result_filename = f'stylized_{job_id}.jpg'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        img_tensor = stylized_image.cpu().squeeze(0).clamp(0, 1)
        transforms.ToPILImage()(img_tensor).save(result_path, quality=90)

        elapsed = time.perf_counter() - t0
        app.logger.info("NST job %s done in %.2fs", job_id, elapsed)
        _set_job(job_id, status='done', result=result_filename, elapsed=round(elapsed, 2))

    except Exception:
        app.logger.exception("NST job %s failed", job_id)
        _set_job(job_id, status='error', error=traceback.format_exc())
    finally:
        del content_tensor, style_tensor, stylized_image
        del content_feats, style_feats, stylized_feats
        gc.collect()


def _set_job(job_id, **kwargs):
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(kwargs)
        while len(_jobs) > MAX_JOBS:
            _jobs.popitem(last=False)


def _enqueue(job_id, content_path, style_path, alpha):
    with _queue_lock:
        _inference_queue.append((job_id, content_path, style_path, alpha))
    _worker_event.set()
    _get_or_start_worker()

# ---------------------------------------------------------------------------
# Model loading — once at startup
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app.logger.info("NST: loading models on device=%s", device)

encoder = VGGEncoder(os.path.join(BASE_DIR, 'vgg_normalised.pth')).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(
    torch.load(os.path.join(BASE_DIR, 'decoder.pth'), map_location=device)
)
encoder.eval()
decoder.eval()
app.logger.info("NST: models loaded")

def allowed_file(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/stylize', methods=['POST'])
def stylize():
    """
    Accepts multipart/form-data with fields:
      content        – image file  (or absent if re-using content_path)
      style          – image file  (or absent if re-using style_path)
      content_path   – filename of a previously uploaded content image
      style_path     – filename of a previously uploaded style image
      alpha          – float 0-1

    Returns JSON: { job_id, content_image, style_image }
    The response is sent immediately; inference happens in a background thread.
    """
    content_filename = None
    style_filename = None

    content_file = request.files.get('content')
    if content_file and content_file.filename:
        if not allowed_file(content_file.filename):
            return jsonify({'error': 'Content image must be PNG or JPG.'}), 400
        content_filename = secure_filename(content_file.filename)
        content_file.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
    else:
        content_filename = request.form.get('content_path', '').strip() or None

    style_file = request.files.get('style')
    if style_file and style_file.filename:
        if not allowed_file(style_file.filename):
            return jsonify({'error': 'Style image must be PNG or JPG.'}), 400
        style_filename = secure_filename(style_file.filename)
        style_file.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
    else:
        style_filename = request.form.get('style_path', '').strip() or None

    if not content_filename or not style_filename:
        return jsonify({'error': 'Please provide both content and style images.'}), 400

    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

    if not os.path.exists(content_path):
        return jsonify({'error': f'Content file not found: {content_filename}'}), 400
    if not os.path.exists(style_path):
        return jsonify({'error': f'Style file not found: {style_filename}'}), 400

    try:
        alpha = float(request.form.get('alpha', 1.0))
        alpha = max(0.0, min(1.0, alpha))
    except (TypeError, ValueError):
        alpha = 1.0

    job_id = str(uuid.uuid4())
    _set_job(job_id, status='queued',
             content_image=content_filename,
             style_image=style_filename)
    _enqueue(job_id, content_path, style_path, alpha)

    app.logger.info("NST: enqueued job %s", job_id)
    return jsonify({
        'job_id': job_id,
        'content_image': content_filename,
        'style_image': style_filename,
    })


@app.route('/status/<job_id>')
def job_status(job_id):
    """Polling endpoint. Returns job state as JSON."""
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(job)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)