"""PyQt6 front end.

Two worker threads keep the window responsive: one runs the generation
pipeline, the other polls Blender for viewport screenshots on its own socket so
the user can watch the scene being assembled.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox,
    QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from . import config
from .blender_link import BlenderLink, port_is_open
from .runner import Runner

STYLE = """
QWidget { background: #16181d; color: #e6e8ec; font-size: 13px; }
QGroupBox {
    border: 1px solid #2a2e37; border-radius: 8px;
    margin-top: 14px; padding: 12px 10px 10px 10px; font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #9aa3b2; }
QPlainTextEdit, QTextEdit, QLineEdit, QSpinBox, QComboBox {
    background: #1e2128; border: 1px solid #2f3440; border-radius: 6px; padding: 6px;
    selection-background-color: #3d6fd4;
}
QPlainTextEdit:focus, QTextEdit:focus, QLineEdit:focus { border: 1px solid #4a7de0; }
QPushButton {
    background: #262b34; border: 1px solid #363d4a; border-radius: 6px;
    padding: 8px 14px; font-weight: 600;
}
QPushButton:hover { background: #2f3540; }
QPushButton:disabled { color: #5c6270; background: #1c1f26; }
QPushButton#generate {
    background: #3d6fd4; border: 1px solid #4a7de0; color: white; font-size: 14px;
    padding: 11px 18px;
}
QPushButton#generate:hover { background: #4a7de0; }
QPushButton#generate:disabled { background: #2a3242; color: #6f7686; }
QPushButton#stop { background: #7a2c34; border: 1px solid #97383f; color: #ffdadd; }
QPushButton#stop:hover { background: #97383f; }
QPushButton[mode="true"] {
    background: #262b34; border: 1px solid #363d4a; padding: 10px;
}
QPushButton[mode="true"]:checked {
    background: #3d6fd4; border: 1px solid #4a7de0; color: white;
}
QProgressBar {
    border: 1px solid #2f3440; border-radius: 6px; background: #1e2128;
    text-align: center; height: 22px;
}
QProgressBar::chunk { background: #3d6fd4; border-radius: 5px; }
QTabWidget::pane { border: 1px solid #2a2e37; border-radius: 8px; top: -1px; }
QTabBar::tab {
    background: #1c1f26; padding: 8px 16px; border: 1px solid #2a2e37;
    border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px;
    margin-right: 2px; color: #9aa3b2;
}
QTabBar::tab:selected { background: #262b34; color: #e6e8ec; }
QLabel#status { color: #9aa3b2; }
QLabel#hint { color: #7d859a; font-size: 12px; }
QLabel#title { font-size: 19px; font-weight: 700; }
QLabel#viewport { background: #0e1013; border: 1px solid #2a2e37; border-radius: 8px; }
QSplitter::handle { background: #2a2e37; }
"""


class PreviewThread(QThread):
    """Polls Blender's viewport so the user can watch the build happen."""

    frame_ready = pyqtSignal(str)

    def __init__(self, port=9876, interval=2.0):
        super().__init__()
        self.port = port
        self.interval = float(interval)
        self._running = True
        self._paths = [
            os.path.join(tempfile.gettempdir(), f"bstudio_preview_{os.getpid()}_{i}.png")
            for i in (0, 1)
        ]
        self._index = 0

    def stop(self):
        self._running = False

    def run(self):
        link = None
        while self._running:
            try:
                if not port_is_open(port=self.port, timeout=0.4):
                    link = None
                    time.sleep(1.0)
                    continue
                if link is None:
                    link = BlenderLink(port=self.port)
                # Alternate files so Qt never reads one mid-write.
                self._index ^= 1
                path = self._paths[self._index]
                link.viewport_screenshot(path, max_size=1100, timeout=25)
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    self.frame_ready.emit(path)
            except Exception:
                # Blender is busy rendering or restarting; just try again.
                if link is not None:
                    link.close()
                link = None
            for _ in range(int(self.interval * 10)):
                if not self._running:
                    break
                time.sleep(0.1)
        if link is not None:
            link.close()


class RunnerThread(QThread):
    log = pyqtSignal(str)
    event = pyqtSignal(str, object)
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)

    def __init__(self, settings, prompt, mode):
        super().__init__()
        self.settings = settings
        self.prompt = prompt
        self.mode = mode
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        runner = Runner(
            self.settings,
            on_log=self.log.emit,
            on_event=lambda kind, payload: self.event.emit(kind, payload),
            on_progress=lambda f, label: self.progress.emit(f, label),
            should_stop=lambda: self._stop,
        )
        self.done.emit(runner.run(self.prompt, self.mode))


class ViewportLabel(QLabel):
    """Aspect-correct image panel that survives window resizing."""

    def __init__(self, placeholder="The Blender viewport will appear here."):
        super().__init__()
        self.setObjectName("viewport")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = None
        self._placeholder = placeholder
        self.setText(placeholder)

    def show_image(self, path):
        image = QImage(path)
        if image.isNull():
            return
        self._pixmap = QPixmap.fromImage(image)
        self._rescale()

    def clear_image(self):
        self._pixmap = None
        self.setText(self._placeholder)

    def _rescale(self):
        if self._pixmap is None:
            return
        self.setPixmap(self._pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = config.load_config()
        self.runner_thread = None
        self.preview_thread = None
        self.result = None
        self.setWindowTitle(config.APP_NAME)
        self.resize(1500, 950)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._start_preview()

    # -- layout -------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel(config.APP_NAME)
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)
        self.model_label = QLabel(f"{config.MODEL}  •  NVIDIA NIM")
        self.model_label.setObjectName("hint")
        header.addWidget(self.model_label)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 1040])
        root.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        bottom.addWidget(self.progress, 1)
        self.open_folder_btn = QPushButton("Open output folder")
        self.open_folder_btn.clicked.connect(self._open_output)
        self.open_folder_btn.setEnabled(False)
        bottom.addWidget(self.open_folder_btn)
        self.open_result_btn = QPushButton("Open result")
        self.open_result_btn.clicked.connect(self._open_result)
        self.open_result_btn.setEnabled(False)
        bottom.addWidget(self.open_result_btn)
        root.addLayout(bottom)

    def _build_left(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(12)

        prompt_box = QGroupBox("What should Blender build?")
        prompt_layout = QVBoxLayout(prompt_box)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "A weathered lighthouse on a rocky outcrop at sunset, waves below, "
            "beam cutting through sea mist"
        )
        self.prompt_edit.setMinimumHeight(110)
        prompt_layout.addWidget(self.prompt_edit)
        hint = QLabel("Describe the scene in plain language. The art direction is "
                      "written for you.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        prompt_layout.addWidget(hint)
        layout.addWidget(prompt_box)

        mode_box = QGroupBox("Output")
        mode_layout = QHBoxLayout(mode_box)
        self.image_btn = QPushButton("Image")
        self.anim_btn = QPushButton("Animation")
        for button in (self.image_btn, self.anim_btn):
            button.setCheckable(True)
            button.setProperty("mode", "true")
            button.setMinimumHeight(42)
            mode_layout.addWidget(button)
        self.image_btn.setChecked(True)
        self.image_btn.clicked.connect(lambda: self._set_mode("image"))
        self.anim_btn.clicked.connect(lambda: self._set_mode("animation"))
        layout.addWidget(mode_box)

        settings_box = QGroupBox("Settings")
        form = QFormLayout(settings_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.res_combo = QComboBox()
        for label, _w, _h in config.RESOLUTIONS:
            self.res_combo.addItem(label)
        saved_res = str(self.settings.get("resolution", "1920x1080"))
        for index in range(self.res_combo.count()):
            if self.res_combo.itemText(index).startswith(saved_res.split()[0]):
                self.res_combo.setCurrentIndex(index)
                break
        form.addRow("Resolution", self.res_combo)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["EEVEE (fast)", "CYCLES (OptiX GPU)"])
        if str(self.settings.get("engine", "EEVEE")).upper().startswith("CYC"):
            self.engine_combo.setCurrentIndex(1)
        form.addRow("Engine", self.engine_combo)

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(8, 1024)
        self.samples_spin.setValue(int(self.settings.get("samples", 64)))
        form.addRow("Samples", self.samples_spin)

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(8, 1200)
        self.frames_spin.setValue(int(self.settings.get("frames", 96)))
        form.addRow("Frames", self.frames_spin)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(6, 60)
        self.fps_spin.setValue(int(self.settings.get("fps", 24)))
        form.addRow("FPS", self.fps_spin)
        layout.addWidget(settings_box)

        advanced = QGroupBox("Model and connection")
        adv_form = QFormLayout(advanced)
        self.key_edit = QLineEdit(
            config.resolve_api_key(self.settings.get("api_key"))
        )
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("nvapi-...  (stored in .env)")
        adv_form.addRow("NVIDIA key", self.key_edit)

        # Nemotron's reasoning control is a boolean, so these are checkboxes and
        # not the low/medium/high combos gpt-oss needed.
        self.director_think = QCheckBox("Let the director think before writing "
                                       "the brief")
        self.director_think.setChecked(
            bool(self.settings.get("director_thinking", True)))
        adv_form.addRow("Director", self.director_think)

        self.builder_think = QCheckBox("Let the builder think between tool calls "
                                      "(slower)")
        self.builder_think.setChecked(
            bool(self.settings.get("builder_thinking", False)))
        adv_form.addRow("Builder", self.builder_think)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(int(self.settings.get("port", 9876)))
        adv_form.addRow("Blender port", self.port_spin)
        layout.addWidget(advanced)

        buttons = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setObjectName("generate")
        self.generate_btn.clicked.connect(self._start_run)
        buttons.addWidget(self.generate_btn, 2)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.clicked.connect(self._stop_run)
        self.stop_btn.setEnabled(False)
        buttons.addWidget(self.stop_btn, 1)
        layout.addLayout(buttons)

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self._set_mode("image")
        return panel

    def _build_right(self):
        self.tabs = QTabWidget()

        self.viewport = ViewportLabel(
            "The Blender viewport appears here once Blender is running."
        )
        live = QWidget()
        live_layout = QVBoxLayout(live)
        live_layout.setContentsMargins(10, 10, 10, 10)
        live_layout.addWidget(self.viewport, 1)
        self.live_hint = QLabel("Live viewport - the Blender window is also open and "
                                "fully interactive.")
        self.live_hint.setObjectName("hint")
        live_layout.addWidget(self.live_hint)
        self.tabs.addTab(live, "Live view")

        self.result_view = ViewportLabel("The finished render appears here.")
        result = QWidget()
        result_layout = QVBoxLayout(result)
        result_layout.setContentsMargins(10, 10, 10, 10)
        result_layout.addWidget(self.result_view, 1)
        self.tabs.addTab(result, "Result")

        self.brief_view = QTextEdit()
        self.brief_view.setReadOnly(True)
        self.brief_view.setPlaceholderText(
            "The art director's brief will appear here once it is written."
        )
        self.tabs.addTab(self.brief_view, "Brief")

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.log_view, "Activity")

        self.code_view = QPlainTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setFont(QFont("Consolas", 10))
        self.code_view.setPlaceholderText("Blender code written by the model appears here.")
        self.tabs.addTab(self.code_view, "Code")
        return self.tabs

    # -- behaviour ----------------------------------------------------------
    def _set_mode(self, mode):
        self.mode = mode
        self.image_btn.setChecked(mode == "image")
        self.anim_btn.setChecked(mode == "animation")
        for widget in (self.frames_spin, self.fps_spin):
            widget.setEnabled(mode == "animation")

    def _collect_settings(self):
        settings = dict(self.settings)
        settings.update({
            "api_key": self.key_edit.text().strip(),
            "resolution": self.res_combo.currentText(),
            "engine": "CYCLES" if self.engine_combo.currentIndex() == 1 else "EEVEE",
            "samples": self.samples_spin.value(),
            "frames": self.frames_spin.value(),
            "fps": self.fps_spin.value(),
            "director_thinking": self.director_think.isChecked(),
            "builder_thinking": self.builder_think.isChecked(),
            "port": self.port_spin.value(),
        })
        return settings

    def _start_preview(self):
        if self.preview_thread is not None:
            self.preview_thread.stop()
            self.preview_thread.wait(2000)
        self.preview_thread = PreviewThread(
            port=int(self.port_spin.value()),
            interval=float(self.settings.get("preview_interval", 2.0)),
        )
        self.preview_thread.frame_ready.connect(self.viewport.show_image)
        self.preview_thread.start()

    def _start_run(self):
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No prompt",
                                "Describe the scene you want before generating.")
            return
        settings = self._collect_settings()
        if not config.resolve_api_key(settings.get("api_key")):
            QMessageBox.warning(
                self, "No API key",
                "Paste your NVIDIA API key into the 'NVIDIA key' field, or set "
                "the NVIDIA_API_KEY environment variable.\n\n"
                "Free keys: https://build.nvidia.com/",
            )
            return

        self.settings = settings
        config.save_config(settings)
        config.save_api_key(settings.get("api_key"))
        self.log_view.clear()
        self.code_view.clear()
        self.brief_view.clear()
        self.result_view.clear_image()
        self.result = None
        self.open_folder_btn.setEnabled(False)
        self.open_result_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tabs.setCurrentIndex(0)
        self._append_log(f"Prompt: {prompt}")
        self._append_log(f"Mode: {self.mode}")

        self._start_preview()
        self.runner_thread = RunnerThread(settings, prompt, self.mode)
        self.runner_thread.log.connect(self._append_log)
        self.runner_thread.event.connect(self._handle_event)
        self.runner_thread.progress.connect(self._handle_progress)
        self.runner_thread.done.connect(self._handle_done)
        self.runner_thread.start()

    def _stop_run(self):
        if self.runner_thread is not None:
            self.runner_thread.request_stop()
            self._append_log("Stop requested; finishing the current step...")
            self.stop_btn.setEnabled(False)

    def _append_log(self, message):
        self.log_view.appendPlainText(message)
        self.status_label.setText(message[:160])

    def _handle_progress(self, fraction, label):
        self.progress.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        if label:
            self.progress.setFormat(f"{label}  -  %p%")

    def _handle_event(self, kind, payload):
        if kind == "brief":
            self.brief_view.setPlainText(payload.get("text", ""))
            self.tabs.setTabText(2, "Brief ●")
        elif kind == "step":
            self.code_view.appendPlainText(
                f"# --- {payload.get('label', 'step')} ---\n{payload.get('code', '')}\n"
            )
        elif kind == "analysis":
            hints = payload.get("hints") or []
            if hints:
                self._append_log("Analysis: " + "; ".join(hints)[:300])
            else:
                self._append_log(
                    "Analysis: framing and exposure look good "
                    f"(luminance {payload.get('mean_luminance', 0):.2f}, "
                    f"contrast {payload.get('contrast', 0):.2f})."
                )
        elif kind == "result_image":
            path = payload.get("path")
            if path and os.path.exists(path):
                self.result_view.show_image(path)
                self.tabs.setCurrentIndex(1)

    def _handle_done(self, result):
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.result = result

        if result.run_dir:
            self.open_folder_btn.setEnabled(True)
        if result.output_path and os.path.exists(result.output_path):
            self.open_result_btn.setEnabled(True)

        if result.cancelled:
            self.progress.setFormat("Cancelled")
            self.status_label.setText("Run cancelled.")
            return
        if result.error:
            self.progress.setFormat("Failed")
            QMessageBox.critical(self, "Generation failed", result.error)
            return

        self.progress.setValue(100)
        self.progress.setFormat("Done  -  %p%")
        if result.output_path and result.output_path.lower().endswith(".png"):
            self.result_view.show_image(result.output_path)
            self.tabs.setCurrentIndex(1)
        self.status_label.setText(f"Saved: {result.output_path}")
        self._append_log(f"Finished. Output: {result.output_path}")

    # -- shell helpers ------------------------------------------------------
    def _open_output(self):
        if self.result and self.result.run_dir:
            self._reveal(self.result.run_dir)

    def _open_result(self):
        if self.result and self.result.output_path:
            self._reveal(self.result.output_path)

    @staticmethod
    def _reveal(path):
        try:
            if os.name == "nt":
                os.startfile(path)  # noqa: S606 - opening the user's own output
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except OSError:
            pass

    def closeEvent(self, event):
        if self.preview_thread is not None:
            self.preview_thread.stop()
            self.preview_thread.wait(2500)
        if self.runner_thread is not None and self.runner_thread.isRunning():
            self.runner_thread.request_stop()
            self.runner_thread.wait(4000)
        current = self._collect_settings()
        config.save_config(current)
        config.save_api_key(current.get("api_key"))
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
