import sys, requests

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

API_URL = "http://127.0.0.1:8000/annotated"


class AnnotatedViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotated PDF Viewer (Red/Green Boxes)")
        self.path = None

        # 상단 컨트롤
        open_btn = QPushButton("PDF 열기")
        open_btn.clicked.connect(self.open_pdf)

        self.page_spin = QSpinBox()
        self.page_spin.setRange(0, 999)
        self.page_spin.setValue(0)
        self.page_spin.setPrefix("page: ")
        self.page_spin.valueChanged.connect(self.refresh)

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.5, 6.0)
        self.zoom_spin.setSingleStep(0.5)
        self.zoom_spin.setValue(2.0)
        self.zoom_spin.setPrefix("zoom: ")
        self.zoom_spin.valueChanged.connect(self.refresh)

        self.cb_labels = QCheckBox("라벨(빨강)")
        self.cb_labels.setChecked(True)
        self.cb_labels.stateChanged.connect(self.refresh)

        self.cb_values = QCheckBox("값(초록)")
        self.cb_values.setChecked(True)
        self.cb_values.stateChanged.connect(self.refresh)

        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(self.page_spin)
        top.addWidget(self.zoom_spin)
        top.addWidget(self.cb_labels)
        top.addWidget(self.cb_values)
        top.addStretch(1)

        # 이미지 표시
        self.label = QLabel("여기에 결과 이미지가 표시됩니다.")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background:#222; color:#ddd;")

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(top)
        layout.addWidget(self.label)
        self.setCentralWidget(root)

    def open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "PDF 선택", "", "PDF Files (*.pdf)")
        if not path:
            return
        self.path = path
        self.refresh()

    def refresh(self):
        if not self.path:
            return
        try:
            with open(self.path, "rb") as f:
                files = {"file": (self.path.split("/")[-1], f, "application/pdf")}
                params = {
                    "page": self.page_spin.value(),
                    "zoom": self.zoom_spin.value(),
                    "labels": "true" if self.cb_labels.isChecked() else "false",
                    "values": "true" if self.cb_values.isChecked() else "false",
                }
                resp = requests.post(API_URL, files=files, params=params, timeout=60)
                resp.raise_for_status()
                pixmap = QPixmap()
                pixmap.loadFromData(resp.content)
                self.label.setPixmap(pixmap)
        except Exception as e:
            self.label.setText(f"요청 실패: {e}")


def main():
    app = QApplication(sys.argv)
    w = AnnotatedViewer()
    w.resize(1000, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
