"""The production window: main prompt in, film out.

The flow the user asked for, in two steps:

1. **Main prompt.** The user gives a strong chat model the seed prompt (the
   "Copy seed prompt" button) and the Persian story, and pastes the main prompt
   it returns here. **Analyse** runs the in-loop model -- Nemotron on NVIDIA's
   endpoint -- over it: parsing, repair if it needs any, moves understood as
   clips, the cast staged and every camera solved (``pipeline.analysis``).
2. **The panel.** What the analysis found, as five tabs, each asking only for
   what the pipeline cannot make itself:

   * **Characters** -- an image and a voice sample for each (the narrator's
     sample too, when there is narration).
   * **Scenes** -- an image for each.
   * **Sentences** -- a recording of every line, recorded here or chosen from a
     file. Each is converted to its speaker's voice (Seed-VC, from the voice
     sample) and lip-synced; narration is placed between the lines, never over
     them, and moves no mouth.
   * **Moves** -- what each character does. A video of someone performing it
     is optional: attached, it drives the character; left blank, the move is
     built from the clips the analysis chose.
   * **Build** -- runs the pipeline and shows the film.

Every attachment is filed by ``pipeline.intake`` exactly where the pipeline's
stages look, so a project the GUI made can also be built from the command line,
and the other way round.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time

from PyQt6.QtCore import (QProcess, QProcessEnvironment, Qt, QThread, QTimer,
                          QUrl, pyqtSignal)
from PyQt6.QtGui import QFont, QGuiApplication, QPixmap, QTextOption
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from . import config

REPO_ROOT = config.PROJECT_DIR
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline import analysis as analysis_mod          # noqa: E402
from pipeline import intake, jsonio, main_prompt, seed  # noqa: E402
from pipeline.project import PROJECTS_ROOT, Project     # noqa: E402

try:                                                    # recording is optional
    from PyQt6.QtMultimedia import (QAudioInput, QAudioOutput,
                                    QMediaCaptureSession, QMediaFormat,
                                    QMediaPlayer, QMediaRecorder)
    MULTIMEDIA = True
except ImportError:                                     # pragma: no cover
    MULTIMEDIA = False

APP_TITLE = "Shahnameh Studio"
EXAMPLE = os.path.join(REPO_ROOT, "examples", "keyumars", "main_prompt.txt")
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
AUDIO_FILTER = "Audio (*.wav *.mp3 *.m4a *.ogg *.flac *.aac *.opus *.webm)"
VIDEO_FILTER = "Video (*.mp4 *.mov *.avi *.mkv *.webm *.m4v)"
STYLES = (("stylized", "Stylized — EEVEE, photographic (recommended)"),
          ("photoreal", "Photoreal — Cycles, slow"),
          ("miniature", "Miniature — flat ink look"))
STAGES = ("a_production", "b_refs", "c_asset", "c_face", "c_body", "c_texture",
          "c_garment", "c_rigged", "d_env", "e_motion", "f_voice", "g_align",
          "h_assemble", "j_render", "j_mouth", "k_edit")
STAGE_WORDS = {
    "a_production": "checking the plan", "b_refs": "reading the images",
    "c_asset": "building 3D characters from the images",
    "c_face": "faces", "c_body": "rigging the characters",
    "c_texture": "textures", "c_garment": "clothing",
    "c_rigged": "publishing the characters",
    "d_env": "building the scenes from their images",
    "e_motion": "animating the moves", "f_voice": "converting the voices",
    "g_align": "lip sync", "h_assemble": "assembling the shots",
    "j_render": "rendering", "j_mouth": "painting the speaking mouths",
    "k_edit": "editing the film",
}

STYLE = """
QWidget { background: #15171c; color: #e7e9ee; font-size: 13px; }
QGroupBox { border: 1px solid #2a2e37; border-radius: 8px; margin-top: 14px;
            padding: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px;
                   color: #9aa3b2; }
QPlainTextEdit, QLineEdit, QComboBox {
    background: #1d2027; border: 1px solid #2f3440; border-radius: 6px;
    padding: 6px; selection-background-color: #3d6fd4; }
QPushButton { background: #262b34; border: 1px solid #363d4a; border-radius: 6px;
              padding: 7px 12px; font-weight: 600; }
QPushButton:hover { background: #2f3540; }
QPushButton:disabled { color: #5c6270; background: #1b1e24; }
QPushButton#primary { background: #3d6fd4; border: 1px solid #4a7de0;
                      color: white; font-size: 14px; padding: 10px 18px; }
QPushButton#primary:hover { background: #4a7de0; }
QPushButton#primary:disabled { background: #2a3242; color: #6f7686; }
QPushButton#rec { background: #5a2a30; border: 1px solid #7a3840; }
QPushButton#rec[recording="true"] { background: #c0392b; color: white; }
QFrame#card { background: #1a1d23; border: 1px solid #2a2e37;
              border-radius: 8px; }
QLabel#muted { color: #8a93a4; }
QLabel#ok { color: #4cc38a; font-weight: 600; }
QLabel#need { color: #e0a84f; font-weight: 600; }
QLabel#optional { color: #7d859a; }
QLabel#title { font-size: 20px; font-weight: 700; }
QLabel#persian { font-size: 17px; }
QLabel#thumb { background: #0f1115; border: 1px dashed #3a404c;
               border-radius: 6px; color: #6f7686; }
QTabWidget::pane { border: 1px solid #2a2e37; border-radius: 8px; top: -1px; }
QTabBar::tab { background: #1b1e24; padding: 8px 16px; border: 1px solid #2a2e37;
               border-bottom: none; border-top-left-radius: 6px;
               border-top-right-radius: 6px; margin-right: 2px; color: #9aa3b2; }
QTabBar::tab:selected { background: #262b34; color: #e7e9ee; }
QProgressBar { border: 1px solid #2f3440; border-radius: 6px; background: #1d2027;
               text-align: center; height: 22px; }
QProgressBar::chunk { background: #3d6fd4; border-radius: 5px; }
"""


def pipeline_python():
    """The interpreter that runs ``python -m pipeline``.

    The pipeline's host steps have run under the base Anaconda interpreter on
    this machine; the GUI itself runs in the ``agent`` env. Either works for the
    host half (the other stages resolve their own envs), so the base one is
    preferred when present only because it is the one that has been exercised.
    """
    candidate = os.path.join(os.path.expanduser("~"), "anaconda3", "python.exe")
    return candidate if os.path.exists(candidate) else sys.executable


def reveal(path):
    try:
        if os.name == "nt":
            os.startfile(path)                      # noqa: S606 - user's file
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except OSError:
        pass


def status_label(state, required=True):
    """A small "attached / needed / optional" label."""
    if state:
        label = QLabel("✓ attached")
        label.setObjectName("ok")
    elif required:
        label = QLabel("● needed")
        label.setObjectName("need")
    else:
        label = QLabel("optional")
        label.setObjectName("optional")
    return label


# -- character image check ------------------------------------------------------
# ``pipeline.image_check`` says whether an attached character image will build
# well (limbs apart, plain background, level camera, even light), in the ml
# env, a couple of seconds per image. Results are kept next to the image with
# the image's size and time, so a replaced image is checked again and an
# unchanged one never is.
VERDICT_STYLE = {"good": ("#58c27d", "Good reference"),
                 "usable": ("#e0b04a", "Usable, could be better"),
                 "poor": ("#e5645a", "Poor reference: regenerate"),
                 "error": ("#9aa3b2", "Not checked")}


def check_path(image):
    return image + ".check.json"


def load_check(image):
    """The saved check for this image, if it is for this version of it."""
    try:
        saved = jsonio.read(check_path(image))
        stat = os.stat(image)
    except (OSError, ValueError):
        return None
    if not saved or saved.get("stamp") != [stat.st_size, int(stat.st_mtime)]:
        return None
    from pipeline import image_check
    if saved.get("version", 1) < image_check.VERSION:
        return None                 # the check itself changed since
    return saved


class ImageChecker:
    """Runs image checks one at a time in the ml env; calls back when done."""

    def __init__(self, parent, done):
        self.parent, self.done = parent, done
        self.queue, self.process, self.current = [], None, None

    def request(self, image, kind):
        if image and all(image != q[0] for q in self.queue) \
                and image != (self.current or (None,))[0]:
            self.queue.append((image, kind))
            self._next()

    def _next(self):
        if self.process is not None or not self.queue:
            return
        from pipeline import envs
        try:
            exe = envs.resolve("ml")
        except envs.EnvError:
            self.queue.clear()
            return
        self.current = self.queue.pop(0)
        image, kind = self.current
        extra, vendor = envs.step_env("ml")
        env = QProcessEnvironment.systemEnvironment()
        for key, value in extra.items():
            env.insert(key, value)
        env.insert("PYTHONPATH", os.pathsep.join([REPO_ROOT] + vendor))
        env.insert("PYTHONIOENCODING", "utf-8")
        self.process = QProcess(self.parent)
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(REPO_ROOT)
        self.process.finished.connect(self._finished)
        self.process.start(exe, ["-m", "pipeline.image_check", image,
                                 "--kind", kind])

    def _finished(self, *_):
        import json
        output = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", "replace")
        image, _kind = self.current
        self.process, self.current = None, None
        result = None
        for line in output.splitlines():
            if line.startswith("IMAGE_CHECK "):
                result = json.loads(line[len("IMAGE_CHECK "):])
        if result is not None:
            stat = os.stat(image)
            result["stamp"] = [stat.st_size, int(stat.st_mtime)]
            jsonio.write(check_path(image), result)
        self.done(image)
        self._next()


def check_label(result):
    """The verdict and its reasons, for a character card."""
    if result is None:
        label = QLabel("Checking the image…")
        label.setObjectName("muted")
        return label
    colour, title = VERDICT_STYLE.get(result.get("verdict"),
                                      VERDICT_STYLE["error"])
    reasons = [f["message"] for f in result.get("findings") or []
               if f.get("level") in ("warn", "fail")]
    html = f"<b style='color:{colour}'>{title}</b>"
    if reasons:
        html += "<ul style='margin:2px 0 0 -24px'>" + "".join(
            f"<li>{r}</li>" for r in reasons) + "</ul>"
    label = QLabel(html)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    return label


# -- threads ------------------------------------------------------------------

class AnalyseThread(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(object, object)            # analysis, error

    def __init__(self, text, api_key, use_model=True):
        super().__init__()
        self.text = text
        self.api_key = api_key
        self.use_model = use_model

    def run(self):
        try:
            chat = None
            if self.use_model:
                chat = analysis_mod.chat_client(self.api_key,
                                                on_log=self.log.emit)
            result = analysis_mod.analyse(self.text, chat=chat,
                                          log=self.log.emit)
            self.done.emit(result, None)
        except Exception as exc:                    # noqa: BLE001 - shown to user
            self.done.emit(None, f"{type(exc).__name__}: {exc}")


# -- widgets ------------------------------------------------------------------

class Thumb(QLabel):
    """An image slot that also takes a dropped file."""

    dropped = pyqtSignal(str)

    def __init__(self, text="drop an image\nor choose one", size=(150, 190)):
        super().__init__(text)
        self.setObjectName("thumb")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(*size)
        self.setAcceptDrops(True)

    def show_file(self, path):
        if path and os.path.exists(path):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self.setPixmap(pixmap.scaled(
                    self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
                return
        self.setPixmap(QPixmap())
        self.setText("drop an image\nor choose one")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.dropped.emit(urls[0].toLocalFile())


class Recorder:
    """One microphone recording at a time, to a WAV in the temp folder."""

    def __init__(self):
        self.session = self.recorder = None
        self.path = None
        if not MULTIMEDIA:
            return
        self.session = QMediaCaptureSession()
        self.input = QAudioInput()
        self.session.setAudioInput(self.input)
        self.recorder = QMediaRecorder()
        self.session.setRecorder(self.recorder)
        media = QMediaFormat()
        media.setFileFormat(QMediaFormat.FileFormat.Wave)
        media.setAudioCodec(QMediaFormat.AudioCodec.Wave)
        self.recorder.setMediaFormat(media)
        self.recorder.setQuality(QMediaRecorder.Quality.HighQuality)

    @property
    def available(self):
        return self.recorder is not None

    def start(self):
        handle, self.path = tempfile.mkstemp(prefix="line_", suffix=".wav")
        os.close(handle)
        os.remove(self.path)
        self.recorder.setOutputLocation(QUrl.fromLocalFile(self.path))
        self.recorder.record()

    def stop(self):
        self.recorder.stop()
        return self.path


class Player:
    def __init__(self):
        self.player = None
        if MULTIMEDIA:
            self.player = QMediaPlayer()
            self.output = QAudioOutput()
            self.player.setAudioOutput(self.output)

    def play(self, path):
        if self.player is None:
            reveal(path)
            return
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()


def scroll_of(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    area.setFrameShape(QFrame.Shape.NoFrame)
    return area


# -- the window -----------------------------------------------------------------

class ProductionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = config.load_config()
        self.project = None
        self.analysis = None
        self.thread = None
        self.process = None
        self.recorder = Recorder()
        self.player = Player()
        self.checker = ImageChecker(
            self, lambda image: QTimer.singleShot(0, self._build_panel))
        self.recording_key = None
        self.recording_button = None
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 920)
        self.setStyleSheet(STYLE)
        self._build()
        self._load_projects()

    # -- layout ----------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        header = QHBoxLayout()
        title = QLabel(APP_TITLE)
        title.setObjectName("title")
        header.addWidget(title)
        header.addSpacing(20)
        header.addWidget(QLabel("Project"))
        self.project_combo = QComboBox()
        self.project_combo.setEditable(True)
        self.project_combo.setMinimumWidth(220)
        self.project_combo.lineEdit().setPlaceholderText("new_project_name")
        header.addWidget(self.project_combo)
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self._open_project)
        header.addWidget(open_btn)
        header.addStretch(1)
        header.addWidget(QLabel("NVIDIA key"))
        self.key_edit = QLineEdit(config.resolve_api_key(
            self.settings.get("api_key")))
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("nvapi-...")
        self.key_edit.setFixedWidth(240)
        header.addWidget(self.key_edit)
        model = QLabel(analysis_mod.MODEL)
        model.setObjectName("muted")
        header.addWidget(model)
        root.addLayout(header)

        self.steps = QTabWidget()
        self.steps.addTab(self._build_prompt_page(), "1  Main prompt")
        self.panel = QTabWidget()
        self.steps.addTab(self.panel, "2  Inputs and build")
        self.steps.setTabEnabled(1, False)
        root.addWidget(self.steps, 1)

        self.status = QLabel("Paste the main prompt and press Analyse.")
        self.status.setObjectName("muted")
        root.addWidget(self.status)

    def _build_prompt_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        tips = QLabel(
            "1. Press <b>Copy seed prompt</b>, paste it into a strong model "
            "(Claude Opus) followed by the Persian story.<br>"
            "2. It returns a <b>main prompt</b> and <b>image prompts</b>. "
            "Generate one image per character and per scene from the image "
            "prompts.<br>3. Paste the main prompt below and press "
            "<b>Analyse</b>.")
        tips.setWordWrap(True)
        layout.addWidget(tips)
        row = QHBoxLayout()
        seed_btn = QPushButton("Copy seed prompt")
        seed_btn.clicked.connect(self._copy_seed)
        row.addWidget(seed_btn)
        save_seed = QPushButton("Save seed prompt…")
        save_seed.clicked.connect(self._save_seed)
        row.addWidget(save_seed)
        example = QPushButton("Load example")
        example.clicked.connect(self._load_example)
        row.addWidget(example)
        open_file = QPushButton("Open main prompt file…")
        open_file.clicked.connect(self._open_prompt_file)
        row.addWidget(open_file)
        row.addStretch(1)
        layout.addLayout(row)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "# FILM\ntitle: ...\n\n# CHARACTERS\n## id\n...\n\n# SCENES\n...\n\n"
            "# SHOTS\n...")
        self.prompt_edit.setFont(QFont("Consolas", 11))
        self.prompt_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        split.addWidget(self.prompt_edit)
        self.analysis_log = QPlainTextEdit()
        self.analysis_log.setReadOnly(True)
        self.analysis_log.setFont(QFont("Consolas", 10))
        self.analysis_log.setPlaceholderText(
            "What the analysis found appears here: problems, repairs, how "
            "each move was understood, where each camera went.")
        split.addWidget(self.analysis_log)
        split.setSizes([820, 480])
        layout.addWidget(split, 1)

        bottom = QHBoxLayout()
        self.use_model = QCheckBox("Use the model (repair + move "
                                   "understanding)")
        self.use_model.setChecked(True)
        bottom.addWidget(self.use_model)
        bottom.addStretch(1)
        self.analyse_btn = QPushButton("Analyse")
        self.analyse_btn.setObjectName("primary")
        self.analyse_btn.clicked.connect(self._analyse)
        bottom.addWidget(self.analyse_btn)
        layout.addLayout(bottom)
        return page

    # -- step 1 ----------------------------------------------------------
    def _copy_seed(self):
        QGuiApplication.clipboard().setText(seed.build_seed_prompt())
        self.status.setText("Seed prompt copied. Paste it into the strong model, "
                            "then paste the Persian story after it.")

    def _save_seed(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save seed prompt",
                                              "seed_prompt.md",
                                              "Markdown (*.md);;Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(seed.build_seed_prompt())
            self.status.setText(f"Seed prompt saved to {path}")

    def _load_example(self):
        with open(EXAMPLE, encoding="utf-8") as handle:
            self.prompt_edit.setPlainText(handle.read())
        if not self.project_combo.currentText().strip():
            self.project_combo.setEditText("keyumars")

    def _open_prompt_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Main prompt", "", "Text (*.txt *.md);;All files (*)")
        if path:
            with open(path, encoding="utf-8") as handle:
                self.prompt_edit.setPlainText(handle.read())

    def _project_name(self):
        name = self.project_combo.currentText().strip().lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")
        if name and not name[0].isalpha():
            name = "p_" + name
        return name

    def _analyse(self):
        text = self.prompt_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No main prompt",
                                "Paste the main prompt first.")
            return
        name = self._project_name()
        if not name:
            QMessageBox.warning(self, "No project name",
                                "Type a project name at the top (letters, "
                                "digits, underscores).")
            return
        key = self.key_edit.text().strip()
        if self.use_model.isChecked() and not config.resolve_api_key(key):
            QMessageBox.warning(self, "No NVIDIA key",
                                "Paste your NVIDIA API key at the top, or "
                                "untick 'Use the model'.")
            return
        config.save_api_key(key)
        self.analysis_log.clear()
        self._log(f"Analysing with {analysis_mod.MODEL}…"
                  if self.use_model.isChecked() else "Analysing (no model)…")
        self.analyse_btn.setEnabled(False)
        self.thread = AnalyseThread(text, key, self.use_model.isChecked())
        self.thread.log.connect(self._log)
        self.thread.done.connect(lambda a, e: self._analysed(name, a, e))
        self.thread.start()

    def _log(self, message):
        self.analysis_log.appendPlainText(message)
        self.status.setText(message[:180])

    def _analysed(self, name, result, error):
        self.analyse_btn.setEnabled(True)
        if error:
            self._log("Failed: " + error)
            QMessageBox.critical(self, "Analysis failed", error)
            return
        self._report(result)
        if result["errors"]:
            QMessageBox.warning(
                self, "The main prompt has problems",
                f"{result['errors']} problem(s) remain after analysis; see the "
                "list on the right. Fix them in the main prompt and analyse "
                "again.")
            return
        try:
            self.project = intake.create(name, result["text"], result)
        except Exception as exc:                    # noqa: BLE001
            QMessageBox.critical(self, "Could not create the project", str(exc))
            return
        if result["text"] != self.prompt_edit.toPlainText().strip():
            self.prompt_edit.setPlainText(result["text"])
            self._log("The main prompt was repaired; the corrected text is now "
                      "in the editor.")
        self.analysis = result
        self._load_projects(select=name)
        self._build_panel()
        self.steps.setTabEnabled(1, True)
        self.steps.setCurrentIndex(1)

    def _report(self, result):
        self._log(f"Done in {result['seconds']} s; model used for: "
                  f"{', '.join(result['llm_used']) or 'nothing'}")
        for problem in result["problems"]:
            self._log(f"[{problem['severity']}] {problem['where']}: "
                      f"{problem['message']}")
        for shot in result["shots"]:
            framing = shot.get("framing") or {}
            self._log(f"{shot['id']}: {framing.get('size')} on "
                      f"{framing.get('subject')}, {framing.get('angle')}, "
                      f"{framing.get('move')} — {len(shot['lines'])} line(s)")
            for member in shot["cast"]:
                move = member["move"]
                self._log(f"    {member['character']}: "
                          f"{' → '.join(c['clip'] for c in move['clips'])}"
                          f"  ({move['source']})"
                          + (f"  “{move['text']}”" if move["text"] else ""))
        for note in result["notes"]:
            self._log("note: " + note)

    # -- projects --------------------------------------------------------
    def _load_projects(self, select=None):
        current = select or self.project_combo.currentText()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for name in Project.list_all():
            if os.path.exists(os.path.join(PROJECTS_ROOT, name, "story",
                                           "analysis.json")):
                self.project_combo.addItem(name)
        self.project_combo.setEditText(current or "")
        self.project_combo.blockSignals(False)

    def _open_project(self):
        name = self._project_name()
        try:
            project = Project.open(name)
        except FileNotFoundError:
            QMessageBox.information(self, "No such project",
                                    f"There is no project '{name}' yet. Paste "
                                    "a main prompt and press Analyse to make "
                                    "it.")
            return
        result = jsonio.read(project.path("story", "analysis.json"),
                             default=None)
        if not result:
            QMessageBox.information(self, "Not analysed",
                                    "This project was not made from a main "
                                    "prompt; open its main prompt and analyse "
                                    "it.")
            return
        self.project, self.analysis = project, result
        self.prompt_edit.setPlainText(result.get("text", ""))
        self.analysis_log.clear()
        self._report(result)
        self._build_panel()
        self.steps.setTabEnabled(1, True)
        self.steps.setCurrentIndex(1)

    # -- step 2: the panel ------------------------------------------------
    def _build_panel(self):
        current = self.panel.currentIndex()
        self.panel.clear()
        state = intake.status(self.project, self.analysis)
        self.state = state
        self.panel.addTab(scroll_of(self._characters_tab(state)),
                          self._tab_title("Characters", state["characters"],
                                          ("image",)))
        self.panel.addTab(scroll_of(self._scenes_tab(state)),
                          self._tab_title("Scenes", state["scenes"],
                                          ("image",)))
        self.panel.addTab(scroll_of(self._sentences_tab(state)),
                          self._tab_title("Sentences", state["sentences"],
                                          ("audio",)))
        self.panel.addTab(scroll_of(self._moves_tab(state)),
                          f"Moves ({len(state['moves'])})")
        self.panel.addTab(self._build_tab(state), "Build")
        if current >= 0:
            self.panel.setCurrentIndex(current)

    @staticmethod
    def _tab_title(name, rows, fields):
        done = sum(1 for r in rows if all(r.get(f) for f in fields))
        return f"{name} ({done}/{len(rows)})"

    def _attach(self, kind, key, path):
        if not path:
            return
        try:
            intake.attach(self.project, kind, key, path)
        except Exception as exc:                    # noqa: BLE001
            QMessageBox.critical(self, "Could not attach", str(exc))
            return
        self.status.setText(f"Attached {os.path.basename(path)} for {key}.")
        QTimer.singleShot(0, self._build_panel)

    def _choose(self, kind, key, title, filter_):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_)
        self._attach(kind, key, path)

    def _clear(self, kind, key):
        intake.detach(self.project, kind, key)
        QTimer.singleShot(0, self._build_panel)

    def _card(self):
        card = QFrame()
        card.setObjectName("card")
        return card

    def _characters_tab(self, state):
        page = QWidget()
        grid = QGridLayout(page)
        grid.setSpacing(12)
        prompts = {c["id"]: c for c in self.analysis["characters"]}
        for index, row in enumerate(state["characters"]):
            card = self._card()
            layout = QHBoxLayout(card)
            thumb = Thumb()
            thumb.show_file(row["image"])
            thumb.dropped.connect(
                lambda p, k=row["id"]: self._attach("character_image", k, p))
            layout.addWidget(thumb)
            body = QVBoxLayout()
            head = QLabel(f"<b>{row['name']}</b>  <span style='color:#9aa3b2'>"
                          f"{row['name_fa']} · {row['kind']} · "
                          f"{prompts[row['id']]['height_m']:.2f} m</span>")
            body.addWidget(head)
            look = QLabel(row["look"])
            look.setWordWrap(True)
            look.setObjectName("muted")
            body.addWidget(look)
            line = QHBoxLayout()
            line.addWidget(QLabel("Image"))
            line.addWidget(status_label(row["image"]))
            choose = QPushButton("Choose image…")
            choose.clicked.connect(lambda _=False, k=row["id"]: self._choose(
                "character_image", k, "Character image", IMAGE_FILTER))
            line.addWidget(choose)
            line.addStretch(1)
            body.addLayout(line)
            if row["image"]:
                result = load_check(row["image"])
                if result is None:
                    self.checker.request(row["image"], row["kind"])
                body.addWidget(check_label(result))
            body.addLayout(self._voice_row(row["id"], row["voice"],
                                           row.get("voice_hint")
                                           or prompts[row["id"]].get("voice")))
            body.addStretch(1)
            layout.addLayout(body, 1)
            grid.addWidget(card, index // 2, index % 2)
        if "narrator_voice" in state:
            card = self._card()
            layout = QVBoxLayout(card)
            layout.addWidget(QLabel("<b>Narrator</b>  <span style='color:#9aa3b2'>"
                                    "voice over the picture; no lips</span>"))
            layout.addLayout(self._voice_row(main_prompt.NARRATOR,
                                             state["narrator_voice"],
                                             "optional: without a sample the "
                                             "narration keeps the recorded "
                                             "voice"))
            grid.addWidget(card, (len(state["characters"]) + 1) // 2 + 1, 0)
        return page

    def _voice_row(self, key, attached, hint):
        row = QVBoxLayout()
        line = QHBoxLayout()
        line.addWidget(QLabel("Voice sample"))
        line.addWidget(status_label(attached, required=False))
        choose = QPushButton("Choose…")
        choose.clicked.connect(lambda _=False, k=key: self._choose(
            "voice_sample", k, "Voice sample", AUDIO_FILTER))
        line.addWidget(choose)
        record = QPushButton("● Record")
        record.setObjectName("rec")
        record.setEnabled(self.recorder.available)
        record.clicked.connect(lambda _=False, k=key, b=record:
                               self._toggle_record("voice_sample", k, b))
        line.addWidget(record)
        if attached:
            play = QPushButton("▶")
            play.clicked.connect(lambda _=False, p=attached: self.player.play(p))
            line.addWidget(play)
        line.addStretch(1)
        row.addLayout(line)
        silent = (hint or "").strip().lower().startswith(("none", "no ",
                                                          "silent"))
        if hint and silent:
            note = QLabel(f"{hint} — no sample needed.")
            note.setObjectName("muted")
            row.addWidget(note)
        elif hint:
            note = QLabel(f"{hint} — 10-25 s of clean speech is best.")
            note.setObjectName("muted")
            note.setWordWrap(True)
            row.addWidget(note)
        return row

    def _scenes_tab(self, state):
        page = QWidget()
        grid = QGridLayout(page)
        grid.setSpacing(12)
        for index, row in enumerate(state["scenes"]):
            card = self._card()
            layout = QHBoxLayout(card)
            thumb = Thumb(size=(260, 150))
            thumb.show_file(row["image"])
            thumb.dropped.connect(
                lambda p, k=row["id"]: self._attach("scene_image", k, p))
            layout.addWidget(thumb)
            body = QVBoxLayout()
            body.addWidget(QLabel(f"<b>{row['name']}</b>"))
            look = QLabel(f"{row['look']}<br><i>{row['light']}</i>")
            look.setWordWrap(True)
            look.setObjectName("muted")
            body.addWidget(look)
            line = QHBoxLayout()
            line.addWidget(status_label(row["image"]))
            choose = QPushButton("Choose image…")
            choose.clicked.connect(lambda _=False, k=row["id"]: self._choose(
                "scene_image", k, "Scene image", IMAGE_FILTER))
            line.addWidget(choose)
            line.addStretch(1)
            body.addLayout(line)
            body.addStretch(1)
            layout.addLayout(body, 1)
            grid.addWidget(card, index // 2, index % 2)
        return page

    def _sentences_tab(self, state):
        page = QWidget()
        layout = QVBoxLayout(page)
        tip = QLabel("Record each line as it should be performed. It is "
                     "converted to the speaker's voice from their sample and "
                     "the mouth is synced to it. Narration plays between the "
                     "lines, never over them.")
        tip.setWordWrap(True)
        tip.setObjectName("muted")
        layout.addWidget(tip)
        names = {c["id"]: c["name"] for c in self.analysis["characters"]}
        names[main_prompt.NARRATOR] = "Narrator"
        shot = None
        for row in state["sentences"]:
            if row["shot"] != shot:
                shot = row["shot"]
                heading = QLabel(f"<b>{shot}</b>")
                layout.addWidget(heading)
            card = self._card()
            grid = QGridLayout(card)
            who = QLabel(f"<b>{names.get(row['speaker'], row['speaker'])}</b>"
                         f"<br><span style='color:#9aa3b2'>{row['emotion']}"
                         + (" · narration" if row["narration"] else "")
                         + "</span>")
            who.setFixedWidth(150)
            grid.addWidget(who, 0, 0, 2, 1)
            persian = QLabel(row["text_fa"])
            persian.setObjectName("persian")
            persian.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
            persian.setAlignment(Qt.AlignmentFlag.AlignRight
                                 | Qt.AlignmentFlag.AlignVCenter)
            persian.setWordWrap(True)
            grid.addWidget(persian, 0, 1)
            gloss = QLabel(row["text_en"])
            gloss.setObjectName("muted")
            gloss.setWordWrap(True)
            grid.addWidget(gloss, 1, 1)
            controls = QHBoxLayout()
            controls.addWidget(status_label(row["audio"]))
            record = QPushButton("● Record")
            record.setObjectName("rec")
            record.setEnabled(self.recorder.available)
            record.clicked.connect(lambda _=False, k=row["line_id"], b=record:
                                   self._toggle_record("line_audio", k, b))
            controls.addWidget(record)
            choose = QPushButton("Choose…")
            choose.clicked.connect(lambda _=False, k=row["line_id"]: self._choose(
                "line_audio", k, "Recording", AUDIO_FILTER))
            controls.addWidget(choose)
            if row["audio"]:
                play = QPushButton("▶")
                play.clicked.connect(lambda _=False, p=row["audio"]:
                                     self.player.play(p))
                controls.addWidget(play)
            box = QWidget()
            box.setLayout(controls)
            grid.addWidget(box, 0, 2, 2, 1)
            grid.setColumnStretch(1, 1)
            layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _moves_tab(self, state):
        page = QWidget()
        layout = QVBoxLayout(page)
        tip = QLabel("Each character's action in each shot. A video of someone "
                     "performing the move is optional: attached, it drives "
                     "the character (GVHMR); left blank, the move is "
                     "generated from its words (MoMask). Film the whole body, "
                     "head to feet, with the phone standing still - on a "
                     "tripod or propped up - not held in the hand.")
        tip.setWordWrap(True)
        tip.setObjectName("muted")
        layout.addWidget(tip)
        names = {c["id"]: c["name"] for c in self.analysis["characters"]}
        for row in state["moves"]:
            card = self._card()
            line = QHBoxLayout(card)
            label = QLabel(f"<b>{row['shot']}</b> · "
                           f"{names.get(row['character'], row['character'])}")
            label.setFixedWidth(260)
            line.addWidget(label)
            text = QLabel((f"“{row['text']}”<br>" if row["text"] else "")
                          + "<span style='color:#9aa3b2'>"
                          + " → ".join(row["clips"])
                          + f" · {row['source']}</span>")
            text.setWordWrap(True)
            line.addWidget(text, 1)
            line.addWidget(status_label(row["video"], required=False))
            choose = QPushButton("Choose video…")
            choose.clicked.connect(lambda _=False, k=row["take"]: self._choose(
                "move_video", k, "Move video", VIDEO_FILTER))
            line.addWidget(choose)
            if row["video"]:
                play = QPushButton("▶")
                play.clicked.connect(lambda _=False, p=row["video"]: reveal(p))
                line.addWidget(play)
                clear = QPushButton("✕")
                clear.clicked.connect(lambda _=False, k=row["take"]:
                                      self._clear("move_video", k))
                line.addWidget(clear)
            layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _build_tab(self, state):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QHBoxLayout()
        form.addWidget(QLabel("Look"))
        self.style_combo = QComboBox()
        for name, label in STYLES:
            self.style_combo.addItem(label, name)
        current = self.project.config.get("style_pack") or "stylized"
        index = next((i for i, (n, _l) in enumerate(STYLES) if n == current), 0)
        self.style_combo.setCurrentIndex(index)
        form.addWidget(self.style_combo)
        form.addSpacing(20)
        self.draft = QCheckBox("Draft (half resolution, much faster)")
        self.draft.setChecked(True)
        form.addWidget(self.draft)
        form.addStretch(1)
        layout.addLayout(form)

        missing = state["missing"]
        if missing:
            need = QLabel("<b>Still needed before building:</b><br>"
                          + "<br>".join(f"• {m}" for m in missing))
            need.setObjectName("need")
            need.setWordWrap(True)
            layout.addWidget(need)
        else:
            ready = QLabel("Everything required is attached. Voice samples and "
                           "move videos you left out are optional.")
            ready.setObjectName("ok")
            layout.addWidget(ready)

        buttons = QHBoxLayout()
        self.build_btn = QPushButton("Build the film")
        self.build_btn.setObjectName("primary")
        self.build_btn.setEnabled(not missing and self.process is None)
        self.build_btn.clicked.connect(self._start_build)
        buttons.addWidget(self.build_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(self.process is not None)
        self.stop_btn.clicked.connect(self._stop_build)
        buttons.addWidget(self.stop_btn)
        self.film_btn = QPushButton("Open film")
        self.film_btn.clicked.connect(self._open_film)
        self.film_btn.setEnabled(bool(self._film_path()))
        buttons.addWidget(self.film_btn)
        folder = QPushButton("Open project folder")
        folder.clicked.connect(lambda: reveal(self.project.root))
        buttons.addWidget(folder)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, len(STAGES))
        self.progress.setFormat("Idle")
        layout.addWidget(self.progress)
        self.build_log = QPlainTextEdit()
        self.build_log.setReadOnly(True)
        self.build_log.setFont(QFont("Consolas", 10))
        self.build_log.setMaximumBlockCount(6000)
        layout.addWidget(self.build_log, 1)
        return page

    # -- recording -------------------------------------------------------
    def _toggle_record(self, kind, key, button):
        if self.recording_key is None:
            self.recorder.start()
            self.recording_key = (kind, key)
            self.recording_button = button
            button.setText("■ Stop")
            button.setProperty("recording", "true")
            button.style().polish(button)
            self.status.setText(f"Recording {key}… press Stop when done.")
            return
        if self.recording_key != (kind, key):
            QMessageBox.information(self, "Already recording",
                                    "Stop the current recording first.")
            return
        path = self.recorder.stop()
        self.recording_key = None
        # The recorder finalises the file asynchronously.
        QTimer.singleShot(700, lambda: self._attach(kind, key, path))

    # -- building --------------------------------------------------------
    def _film_path(self):
        if self.project is None:
            return None
        pack = self.project.config.get("style_pack") or "stylized"
        path = self.project.path("film", pack, "final.mp4")
        return path if os.path.exists(path) else None

    def _start_build(self):
        pack = self.style_combo.currentData()
        config_now = self.project.config
        config_now["style_pack"] = pack
        self.project.save_config(config_now)
        args = ["-m", "pipeline", "run", self.project.name, "--style", pack]
        if self.draft.isChecked():
            args += ["--scale", "0.5"]
        self.build_log.clear()
        self.build_log.appendPlainText("python " + " ".join(args))
        self.progress.setValue(0)
        self.progress.setFormat("Starting…")
        self.process = QProcess(self)
        self.process.setWorkingDirectory(REPO_ROOT)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_build)
        self.process.finished.connect(self._build_finished)
        self.process.start(pipeline_python(), args)
        self.started = time.time()
        self.build_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _read_build(self):
        data = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        for line in data.splitlines():
            self.build_log.appendPlainText(line)
            stage = line.split(" ", 1)[0].strip()
            if stage in STAGES:
                index = STAGES.index(stage)
                self.progress.setValue(index)
                self.progress.setFormat(
                    f"{STAGE_WORDS.get(stage, stage)}  ({index + 1}/"
                    f"{len(STAGES)})")

    def _build_finished(self, code, _status):
        minutes = (time.time() - self.started) / 60.0
        ok = code == 0 and bool(self._film_path())
        self.progress.setValue(len(STAGES) if ok else self.progress.value())
        self.progress.setFormat(f"Done in {minutes:.1f} min" if ok
                                else f"Stopped (exit {code})")
        self.process = None
        self.build_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.film_btn.setEnabled(bool(self._film_path()))
        if ok:
            self.status.setText(f"Film ready: {self._film_path()}")
        else:
            self.status.setText("The build stopped; the log says where and "
                                "why. Fix it and build again: finished work "
                                "is cached.")

    def _stop_build(self):
        if self.process is not None:
            self.process.kill()

    def _open_film(self):
        path = self._film_path()
        if path:
            reveal(path)

    def closeEvent(self, event):
        if self.process is not None:
            answer = QMessageBox.question(
                self, "A build is running",
                "Stop the build and close? Finished stages are kept.")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process.kill()
        settings = dict(self.settings)
        config.save_config(settings)
        config.save_api_key(self.key_edit.text().strip())
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    window = ProductionWindow()
    window.show()
    sys.exit(app.exec())
