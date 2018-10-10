# -*- coding: utf-8 -*-
"""
# @Time    : 6/10/18 5:56 PM
# @Author  : Heng Guo
# @File    : UI.py
"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import UIFunction as UF


class Root(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(300, 400)
        self.setFixedSize(300, 400)
        UF.center(self)
        self.setWindowTitle('Topic Cluster')
        self.statusBar().showMessage('Current no model')
        self.wig = QWidget(self)
        self.setCentralWidget(self.wig)
        self.create_logo()
        self.create_button()

    def create_logo(self):
        logo = QLabel(self)
        pixmap = QPixmap('icon.png')
        pixmap = pixmap.scaledToHeight(120)
        logo.setPixmap(pixmap)
        logo.setGeometry(0, 0, 300, 120)
        logo.setAlignment(Qt.AlignCenter)

    def create_button(self):
        button_wig = QWidget(self.wig)
        button_wig.setGeometry(60, 120, 180, 240)
        grid = QGridLayout()
        grid.setSpacing(5)
        button_wig.setLayout(grid)

        button1 = QPushButton('Train Model')
        button2 = QPushButton('Load Model')
        button3 = QPushButton('Updata Model')
        button4 = QPushButton('Save Model')
        button5 = QPushButton('View Model')

        grid.addWidget(button1, 0, 0)
        grid.addWidget(button2, 1, 0)
        grid.addWidget(button3, 2, 0)
        grid.addWidget(button4, 3, 0)
        grid.addWidget(button5, 4, 0)

        button1.clicked.connect(lambda : UF.choose(self))
        button2.clicked.connect(lambda : UF.choose(self))
        button3.clicked.connect(lambda : UF.choose(self))
        button4.clicked.connect(lambda : UF.choose(self))
        button5.clicked.connect(lambda : UF.choose(self))

    # esc key to quit
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

class FileWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.resize(400,60)
        self.path =''
        self.initUI()

    def initUI(self):
        button = QPushButton('...')
        self.buttonbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonbox.rejected.connect(self.reject)
        self.buttonbox.accepted.connect(self.accept)
        self.le = QLineEdit()


        Hlayout = QHBoxLayout()
        Vlayout = QVBoxLayout()
        Vlayout.addLayout(Hlayout)
        self.setLayout(Vlayout)
        Hlayout.addWidget(self.le)
        Hlayout.addWidget(button)
        Vlayout.addWidget(self.buttonbox)

        button.clicked.connect(self.choose_dir)

    def choose_dir(self):
        self.path = QFileDialog.getExistingDirectory(self, 'Select a folder:', '..', QFileDialog.ShowDirsOnly)
        self.le.setText(self.path)

class TrainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.resize(300,400)
        self.setWindowTitle('Choose model and features')
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        f_layout = QFormLayout()
        v_layout = QVBoxLayout()

        self.m_bt = QComboBox()
        self.m_bt.addItems(['LDA','LSI','HDP'])
        # m_bt.addItem('LSI')
        # m_bt.addItem('HDP')
        l = QLabel('model:')
        self.m_bt.currentIndexChanged.connect(self.choose_model)
        self.le1 = QLineEdit()
        self.le1.setText('1000')
        self.le2 = QLineEdit()
        self.le2.setText('20')
        f_layout.addRow(l,self.m_bt)
        f_layout.addRow('iteration times',self.le1)
        f_layout.addRow('topic numbers:',self.le2)

        groupbox = QGroupBox('Function:')
        vbox = QVBoxLayout()
        self.c1 = QCheckBox('ngram')
        self.c2 = QCheckBox('lemmatization')
        self.c3 = QCheckBox('stop_words')
        self.c4 = QCheckBox('tfidf')
        vbox.addWidget(self.c1)
        vbox.addWidget(self.c2)
        vbox.addWidget(self.c3)
        vbox.addWidget(self.c4)
        groupbox.setLayout(vbox)

        self.button = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button.rejected.connect(self.reject)
        self.button.accepted.connect(self.accept)
        main_layout.addLayout(f_layout)
        v_layout.addWidget(groupbox)
        v_layout.addWidget(self.button)
        main_layout.addLayout(v_layout)
        main_layout.setStretch(0,1)
        main_layout.setStretch(1,2)
        self.setLayout(main_layout)

    def choose_model(self, i):
        if i == 1:
            self.le1.setText('can`t set')
            if self.le2.text() == '' or self.le2.text() == 'can`t set':
                self.le2.setText('20')
            self.le1.setEnabled(False)
            self.le2.setEnabled(True)
        elif i == 2:
            self.le2.setText('can`t set')
            if self.le1.text() == '' or self.le1.text() == 'can`t set':
                self.le1.setText('1000')
            self.le2.setEnabled(False)
            self.le1.setEnabled(True)
        else:
            self.le1.setEnabled(True)
            self.le2.setEnabled(True)
            if self.le1.text() == '' or self.le1.text() == 'can`t set':
                self.le1.setText('1000')
            if self.le2.text() == '' or self.le2.text() == 'can`t set':
                self.le2.setText('20')

class ProgressWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(300,50)
        self.initUI()

    def initUI(self):
        self.pb = QProgressBar()
        self.pb.setMinimum(0)
        self.pb.setMaximum(0)
        self.label = QLabel('Start training, it may take few minutes')
        v_layout = QVBoxLayout()
        v_layout.addWidget(self.label)
        v_layout.addWidget(self.pb)
        self.setLayout(v_layout)

class ViewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(600,800)
        self.initUI()

    def initUI(self):

        self.topic_btn = QSpinBox()
        self.topic_btn.setRange(1,UF.MODEL.num_topics)
        self.topic_btn.setPrefix('topic ')
        self.topic_btn.valueChanged.connect(lambda :self.choose_topic(self.topic_btn.value()))

        self.topic_id = 0
        self.page = 0
        self.page_num = 0

        lb1 = QLabel('key words:')
        lb2 = QLabel('sentence:')
        lb3 = QLabel('document:')
        self.topic_lb = QTextBrowser()
        self.sent_tb = QTextBrowser()
        self.doc_tb = QTextBrowser()

        self.uppage = QPushButton('<<')
        self.downpage = QPushButton('>>')
        self.page_sp = QSpinBox()

        self.uppage.setEnabled(False)
        self.choose_topic(1)

        self.uppage.clicked.connect(lambda :self.choose_file(self.page-1))
        self.downpage.clicked.connect(lambda :self.choose_file(self.page+1))

        self.page_sp.valueChanged.connect(lambda :self.choose_file(self.page_sp.value()))

        grid = QGridLayout()
        grid.addWidget(self.topic_btn,0,0)
        grid.addWidget(lb1,1,0)
        grid.addWidget(self.topic_lb,1,1,2,4)
        grid.addWidget(lb2,3,0)
        grid.addWidget(self.sent_tb,3,1,3,4)
        grid.addWidget(lb3,6,0)
        grid.addWidget(self.doc_tb,6,1,5,4)
        grid.addWidget(self.uppage,11,1)
        grid.addWidget(self.page_sp,11,2)
        grid.addWidget(self.downpage,11,3)
        self.setLayout(grid)

    def choose_topic(self, value):
        self.topic_id = value - 1
        self.topic_lb.setText(UF.MODEL.topic_key.iloc[self.topic_id, 1])
        self.sent_tb.setText(UF.MODEL.topic_sent.iloc[self.topic_id, 1])
        self.doc_tb.setText(UF.find_doc(UF.MODEL, self.topic_id, 0))
        self.page_num = len(UF.MODEL.topic_doc.iloc[self.topic_id, 1].split(','))
        self.page_sp.setRange(1, self.page_num)
        self.page_sp.setSuffix('/{}'.format(self.page_num))

    def choose_file(self, value):
        self.page = value
        self.page_sp.setValue(self.page)
        self.doc_tb.setText(UF.find_doc(UF.MODEL, self.topic_id, self.page - 1))
        if self.page > 1:
            self.uppage.setEnabled(True)
        else:
            self.uppage.setEnabled(False)
        if self.page == self.page_num:
            self.downpage.setEnabled(False)
        else:
            self.downpage.setEnabled(True)
