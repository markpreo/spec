import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QScrollArea,
    QPlainTextEdit,
)


# ============================================================
# ИМПОРТ ВАШЕЙ ФУНКЦИИ
# ============================================================


from avantes.sourse.scripts.after_shot_analysys import visualize

# ============================================================
# СПИСОК СПЕКТРАЛЬНЫХ ЛИНИЙ
# ============================================================

LINES = [
    {
        "name": "CIII, OII group",
        "waves": [464.728, 464.916, 465.025, 465.147],
        "gaps": [0.4, 0.4, 0.4, 0.4],
    },
    {
        "name": "OII group",
        "waves": [434.740, 435.139],
        "gaps": [0.4, 0.4],
    },
    {
        "name": "CII group",
        "waves": [657.8, 658.28],
        "gaps": [0.4, 0.4],
    },
    {
        "name": "CrI",
        "waves": [425.4331, 427.4806, 428.9733],
        "gaps": [0.4, 0.4, 0.4],
    },
    {
        "name": "N II",
        "waves": [568.125],
        "gaps": [1.0],
    },
    {
        "name": "Dgamma",
        "waves": [434.0625],
        "gaps": [0.4],
    },
    {
        "name": "Dbeta",
        "waves": [485.9375],
        "gaps": [0.4],
    },
]


# ============================================================
# ОСНОВНОЕ ОКНО
# ============================================================

class SpectrumWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Спектральный анализ")
        self.resize(850, 650)

        self.checkbox_by_name = {}

        self.create_interface()

    def create_interface(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ----------------------------------------------------
        # Выбор файла
        # ----------------------------------------------------

        file_group = QGroupBox("Выбор файла")
        file_layout = QHBoxLayout(file_group)

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText(
            "Путь к файлу спектра .STR8"
        )

        browse_button = QPushButton("Выбрать файл")
        browse_button.clicked.connect(self.choose_file)

        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(browse_button)

        main_layout.addWidget(file_group)

        # ----------------------------------------------------
        # Выбор линий
        # ----------------------------------------------------

        lines_group = QGroupBox("Выбор спектральных линий")
        lines_layout = QVBoxLayout(lines_group)

        buttons_layout = QHBoxLayout()

        select_all_button = QPushButton("Выбрать все")
        select_all_button.clicked.connect(self.select_all)

        clear_all_button = QPushButton("Снять все")
        clear_all_button.clicked.connect(self.clear_all)

        buttons_layout.addWidget(select_all_button)
        buttons_layout.addWidget(clear_all_button)
        buttons_layout.addStretch()

        lines_layout.addLayout(buttons_layout)

        # Область с прокруткой
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for index, line in enumerate(LINES):
            line_name = line["name"]
            waves = line["waves"]

            waves_text = ", ".join(
                self.format_number(wave)
                for wave in waves
            )

            checkbox_text = (
                f"{line_name}: {waves_text} нм"
            )

            checkbox = QCheckBox(checkbox_text)
            checkbox.stateChanged.connect(
                self.update_selection_info
            )

            self.checkbox_by_name[line_name] = checkbox

            row = index // 2
            column = index % 2

            scroll_layout.addWidget(
                checkbox,
                row,
                column
            )

        scroll_area.setWidget(scroll_widget)
        lines_layout.addWidget(scroll_area)

        main_layout.addWidget(lines_group, stretch=1)

        # ----------------------------------------------------
        # Информация о выбранных параметрах
        # ----------------------------------------------------

        selection_group = QGroupBox("Текущий выбор")
        selection_layout = QVBoxLayout(selection_group)

        self.selection_text = QPlainTextEdit()
        self.selection_text.setReadOnly(True)
        self.selection_text.setMaximumHeight(120)

        selection_layout.addWidget(self.selection_text)

        main_layout.addWidget(selection_group)

        # ----------------------------------------------------
        # Кнопки запуска и сохранения
        # ----------------------------------------------------

        bottom_layout = QHBoxLayout()

        self.run_button = QPushButton("Построить график")
        self.run_button.setMinimumHeight(36)
        self.run_button.clicked.connect(
            self.run_visualization
        )

        export_button = QPushButton(
            "Сохранить параметры в TXT"
        )
        export_button.setMinimumHeight(36)
        export_button.clicked.connect(
            self.export_to_txt
        )

        exit_button = QPushButton("Выход")
        exit_button.setMinimumHeight(36)
        exit_button.clicked.connect(self.close)

        bottom_layout.addWidget(self.run_button)
        bottom_layout.addWidget(export_button)
        bottom_layout.addStretch()
        bottom_layout.addWidget(exit_button)

        main_layout.addLayout(bottom_layout)

        # Строка состояния
        self.statusBar().showMessage(
            "Выберите файл и спектральные линии"
        )

    # ========================================================
    # РАБОТА С ФАЙЛОМ
    # ========================================================

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл спектра",
            "",
            "STR8 files (*.STR8);;Все файлы (*.*)"
        )

        if file_path:
            self.file_edit.setText(file_path)
            self.statusBar().showMessage(
                f"Выбран файл: {Path(file_path).name}"
            )

    # ========================================================
    # РАБОТА С ЧЕКБОКСАМИ
    # ========================================================

    def select_all(self):
        for checkbox in self.checkbox_by_name.values():
            checkbox.setChecked(True)

    def clear_all(self):
        for checkbox in self.checkbox_by_name.values():
            checkbox.setChecked(False)

    def get_selected_data(self):
        """
        Возвращает:

            curr_waves
            gaps
            selected_names
        """

        curr_waves = []
        gaps = []
        selected_names = []

        # Порядок будет таким же, как в списке LINES
        for line in LINES:
            line_name = line["name"]
            checkbox = self.checkbox_by_name[line_name]

            if checkbox.isChecked():
                selected_names.append(line_name)
                curr_waves.extend(line["waves"])
                gaps.extend(line["gaps"])

        return curr_waves, gaps, selected_names

    def update_selection_info(self):
        curr_waves, gaps, selected_names = (
            self.get_selected_data()
        )

        if not curr_waves:
            self.selection_text.setPlainText(
                "Ничего не выбрано"
            )
            return

        lines_text = ", ".join(selected_names)

        waves_text = ", ".join(
            self.format_number(wave)
            for wave in curr_waves
        )

        gaps_text = ", ".join(
            self.format_number(gap)
            for gap in gaps
        )

        result = (
            f"Выбранные линии:\n"
            f"{lines_text}\n\n"
            f"curr_waves:\n"
            f"{waves_text}\n\n"
            f"gaps:\n"
            f"{gaps_text}"
        )

        self.selection_text.setPlainText(result)

    # ========================================================
    # ЗАПУСК ФУНКЦИИ VISUALIZE
    # ========================================================

    def run_visualization(self):
        data_directory = self.file_edit.text().strip()

        if not data_directory:
            QMessageBox.warning(
                self,
                "Файл не выбран",
                "Сначала выберите файл спектра."
            )
            return

        if not Path(data_directory).is_file():
            QMessageBox.critical(
                self,
                "Ошибка",
                "Указанный файл не существует."
            )
            return

        curr_waves, gaps, selected_names = (
            self.get_selected_data()
        )

        if not curr_waves:
            QMessageBox.warning(
                self,
                "Линии не выбраны",
                "Выберите хотя бы одну спектральную линию."
            )
            return

        try:
            self.run_button.setEnabled(False)
            self.statusBar().showMessage(
                "Выполняется обработка..."
            )

            # =================================================
            # ЗДЕСЬ ВЫЗЫВАЕТСЯ ВАША ФУНКЦИЯ
            # =================================================

            visualize(
                data_directory=data_directory,
                curr_waves=curr_waves,
                gaps=gaps
            )

            # =================================================

            self.statusBar().showMessage(
                "Обработка завершена"
            )

            QMessageBox.information(
                self,
                "Готово",
                "Обработка файла завершена."
            )

        except Exception as error:
            self.statusBar().showMessage(
                "Ошибка при обработке"
            )

            QMessageBox.critical(
                self,
                "Ошибка",
                f"{type(error).__name__}: {error}"
            )

        finally:
            self.run_button.setEnabled(True)

    # ========================================================
    # СОХРАНЕНИЕ В TXT
    # ========================================================

    def export_to_txt(self):
        data_directory = self.file_edit.text().strip()

        curr_waves, gaps, selected_names = (
            self.get_selected_data()
        )

        if not data_directory:
            QMessageBox.warning(
                self,
                "Файл не выбран",
                "Сначала выберите файл."
            )
            return

        if not curr_waves:
            QMessageBox.warning(
                self,
                "Линии не выбраны",
                "Выберите хотя бы одну линию."
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить параметры",
            "",
            "TXT files (*.txt);;Все файлы (*.*)"
        )

        if not save_path:
            return

        if not save_path.lower().endswith(".txt"):
            save_path += ".txt"

        try:
            with open(save_path, "w", encoding="utf-8") as file:
                file.write(
                    "Параметры спектрального анализа\n"
                )
                file.write("=" * 45 + "\n\n")

                file.write("Файл:\n")
                file.write(f"{data_directory}\n\n")

                file.write("Выбранные линии:\n")
                for line_name in selected_names:
                    file.write(f"- {line_name}\n")

                file.write("\nДлины волн:\n")
                file.write(
                    ", ".join(
                        self.format_number(wave)
                        for wave in curr_waves
                    )
                )

                file.write("\n\nGaps:\n")
                file.write(
                    ", ".join(
                        self.format_number(gap)
                        for gap in gaps
                    )
                )

                file.write("\n")

            QMessageBox.information(
                self,
                "Готово",
                "Параметры сохранены в TXT-файл."
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Ошибка сохранения",
                str(error)
            )

    # ========================================================
    # ФОРМАТИРОВАНИЕ ЧИСЕЛ
    # ========================================================

    @staticmethod
    def format_number(value):
        return f"{value:g}"


# ============================================================
# ЗАПУСК ПРОГРАММЫ
# ============================================================

def main():
    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = SpectrumWindow()
    window.show()

    sys.exit(app.exec())



if __name__ == "__main__":
    main()
