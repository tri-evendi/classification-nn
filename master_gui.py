import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from pglcm import *
from klasifikasi import *
import os
import datetime


class MyFirstGUI:

    def __init__(self, master):
        global file_citra
        global in_glcm
        global hasil_glcm
        global fitur_0, fitur_45, fitur_90, fitur_135
        self.master = master
        master.title(
            "KLASIFIKASI KEMATANGAN BUAH MANGGA MENGGUNAKAN METODE GLCM DAN BACKPROPAGATION"
        )
        master.geometry("850x650")
        master.configure(bg='#bca674')

        self.lbNama = tk.Label(master,
                               text='nama file',
                               anchor='center',
                               bg='#e6dcbc',
                               fg='#201314')
        self.lbNama.place(x=30, y=65, width=260, height=30)
        self.lbNama.configure(font=("Helvetica", 12))

        self.btPilih = tk.Button(master,
                                 text="PILIH\n GAMBAR",
                                 bd=1,
                                 cursor="hand2",
                                 command=self.fungsi_open,
                                 bg='#e6dcbc',
                                 fg='#201314')
        self.btPilih.place(x=30, y=105, width=110, height=60)
        self.btPilih.configure(font=("Helvetica", 10, 'bold'))

        self.btRun = tk.Button(master,
                               text="RUN",
                               bd=1,
                               cursor="hand2",
                               command=self.fungsi_proses,
                               bg='#e6dcbc',
                               fg='#201314')
        self.btRun.place(x=180, y=105, width=110, height=60)
        self.btRun.configure(font=("Helvetica", 10, 'bold'))

        self.btReset = tk.Button(master,
                                 text="RESET",
                                 bd=1,
                                 cursor="hand2",
                                 command=self.fungsi_reset,
                                 bg='#e6dcbc',
                                 fg='#201314')
        self.btReset.place(x=60, y=180, width=200, height=40)
        self.btReset.configure(font=("Helvetica", 10, 'bold'))

        self.citra_test = tk.Label(master, bg='#e6dcbc')
        self.citra_test.place(x=370, y=60, width=300, height=180)

        self.citra_glcm = tk.Label(master, bg='#e6dcbc')
        # self.citra_gray.place(x=595, y=80, width=180, height=110)

        # Button
        #frame 3: hasil glcm

        self.frame3 = tk.Frame(master, bg='#947147')
        self.frame3.place(x=20, y=260, width=450, height=250)

        #------------Fitur GLCM-------------
        self.lbKontras = tk.Label(self.frame3,
                                  text='CONTRAST',
                                  anchor='w',
                                  bg='#947147',
                                  fg='white')
        self.lbKontras.place(x=10, y=22, width=130, height=35)
        self.lbKontras.configure(font=("Helvetica", 12))

        self.lbhKontras = tk.Label(self.frame3, anchor='e', bg='white')
        self.lbhKontras.place(x=150, y=22, width=280, height=30)
        self.lbhKontras.configure(font=("Helvetica", 12))

        self.lbHomog = tk.Label(self.frame3,
                                text='HOMOGENEITY',
                                anchor='w',
                                bg='#947147',
                                fg='white')
        self.lbHomog.place(x=10, y=62, width=130, height=20)
        self.lbHomog.configure(font=("Helvetica", 12))

        self.lbhHomog = tk.Label(self.frame3, anchor='e', bg='white')
        self.lbhHomog.place(x=150, y=62, width=280, height=30)
        self.lbhHomog.configure(font=("Helvetica", 12))

        self.lbEnergy = tk.Label(self.frame3,
                                 text='ENERGY',
                                 anchor='w',
                                 bg='#947147',
                                 fg='white')
        self.lbEnergy.place(x=10, y=102, width=130, height=20)
        self.lbEnergy.configure(font=("Helvetica", 12))

        self.lbhEnergy = tk.Label(self.frame3, anchor='e', bg='white')
        self.lbhEnergy.place(x=150, y=102, width=280, height=30)
        self.lbhEnergy.configure(font=("Helvetica", 12))

        self.lbCor = tk.Label(self.frame3,
                              text='CORRELATION',
                              anchor='w',
                              bg='#947147',
                              fg='white')
        self.lbCor.place(x=10, y=142, width=130, height=20)
        self.lbCor.configure(font=("Helvetica", 12))

        self.lbhCor = tk.Label(self.frame3, anchor='e', bg='white')
        self.lbhCor.place(x=150, y=142, width=280, height=30)
        self.lbhCor.configure(font=("Helvetica", 12))

        self.lbEnt = tk.Label(self.frame3,
                              text='Entropy',
                              anchor='w',
                              bg='#947147',
                              fg='white')
        self.lbEnt.place(x=10, y=182, width=130, height=20)
        self.lbEnt.configure(font=("Helvetica", 12))

        self.lbhEnt = tk.Label(self.frame3, anchor='e', bg='white')
        self.lbhEnt.place(x=150, y=182, width=280, height=30)
        self.lbhEnt.configure(font=("Helvetica", 12))

        self.lbGLCM = tk.Label(self.frame3,
                               text='GLCM',
                               anchor='center',
                               bg='#947147',
                               fg='white')
        self.lbGLCM.place(x=0, y=215, width=450, height=30)
        self.lbGLCM.configure(font=("Helvetica", 14, 'bold'))

        #frame 4: hasil backpro

        self.frame4 = tk.Frame(master, bg='#947147')
        self.frame4.place(x=505, y=260, width=265, height=220)

        self.lbBPNN = tk.Label(self.frame4,
                               text='HASIL',
                               anchor='center',
                               bg='#947147',
                               fg='white')
        self.lbBPNN.place(x=0, y=25, width=180, height=30)
        self.lbBPNN.configure(font=("Helvetica", 14, 'bold'))

        self.hsBPNN = tk.Label(self.frame4, anchor='w', bg='white')
        self.hsBPNN.place(x=10, y=65, width=200, height=60)
        self.hsBPNN.configure(font=("Helvetica", 12))

        # Frame for Treeview (Table Sudut)
        self.frame5 = tk.Frame(master, bg='#947147')
        self.frame5.place(x=20, y=520, width=750, height=120)

        self.table_sudut = ttk.Treeview(self.frame5,
                                        columns=("Angle", "Contrast",
                                                 "Homogeneity", "Energy",
                                                 "Correlation", "Entropy"),
                                        show='headings')
        self.table_sudut.heading("Angle", text="Angle")
        self.table_sudut.heading("Contrast", text="Contrast")
        self.table_sudut.heading("Homogeneity", text="Homogeneity")
        self.table_sudut.heading("Energy", text="Energy")
        self.table_sudut.heading("Correlation", text="Correlation")
        self.table_sudut.heading("Entropy", text="Entropy")

        self.table_sudut.column("Angle", width=80)
        self.table_sudut.column("Contrast", width=120)
        self.table_sudut.column("Homogeneity", width=120)
        self.table_sudut.column("Energy", width=120)
        self.table_sudut.column("Correlation", width=120)
        self.table_sudut.column("Entropy", width=120)

        self.table_sudut.pack(fill=tk.BOTH, expand=True)

    def greet(self):
        print("Greetings!")

    def fungsi_reset(self):
        self.lbNama.configure(text="")
        self.citra_test.configure(image='')
        self.citra_glcm.configure(image='')
        self.hsBPNN.configure(text="")

    # def fungsi_open(self):
    #     global file_citra
    #     print("-----------------KLASIFIKASI BARU---------------------")
    #     self.nama_file =  tk.filedialog.askopenfilename(initialdir = "/dataset",title = "Select Image",filetypes = (("all files","*.*"),("png files","*.png"),("jpg files","*.jpg")))
    #     self.citra_mangga = Image.open(self.nama_file)
    #     self.citra_mangga = ImageTk.PhotoImage(self.citra_mangga)
    #     self.lbNama.configure(text = self.nama_file)
    #     self.citra_test.configure(image=self.citra_mangga)
    #     self.citra_test.image=self.citra_mangga
    #     print (self.nama_file)
    #     file_citra = self.nama_file
    #     return file_citra

    def fungsi_open(self):
        global file_citra
        print("-----------------KLASIFIKASI BARU---------------------")
        self.nama_file = tk.filedialog.askopenfilename(
            initialdir="/dataset",
            title="Select Image",
            filetypes=(("all files", "*.*"), ("png files", "*.png"),
                       ("jpg files", "*.jpg")))
        print("Nama file : ", self.nama_file)
        # Load the image
        self.citra_mangga = Image.open(self.nama_file)
        width, height = self.citra_mangga.size
        print("Ukuran citra : ", width, "x", height)
        # Calculate center cropping region
        target_aspect_ratio = 4 / 3
        image_aspect_ratio = width / height

        if image_aspect_ratio > target_aspect_ratio:  # Image is too wide
            crop_width = height * target_aspect_ratio
            crop_height = height
        else:  # Image is too tall or aspect ratio matches
            crop_width = width
            crop_height = width / target_aspect_ratio

        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        cropped_image = self.citra_mangga.crop((left, top, right, bottom))
        # Get the filename and extension
        filename, file_extension = os.path.splitext(self.nama_file)
        # Construct the save directory
        project_directory = os.getcwd()
        sample_folder = "sample"
        save_directory = os.path.join(project_directory, sample_folder)
        # Create the 'sample' folder if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        if os.path.exists(save_directory):
            print("Sample folder created successfully")
        # Save the cropped image
        now = datetime.datetime.now()
        new_filename = f"cropped_{now.strftime('%Y%m%d_%H%M%S')}{file_extension}"
        print("New filename : ", new_filename)
        save_path = os.path.join(save_directory, new_filename)
        cropped_image.save(save_path)
        self.citra_mangga = ImageTk.PhotoImage(cropped_image)
        self.lbNama.configure(text=save_path)
        self.citra_test.configure(image=self.citra_mangga)
        self.citra_test.image = self.citra_mangga
        file_citra = save_path
        return file_citra

    def fungsi_proses(self):
        global file_citra
        # Call the function to extract the GLCM features
        self.fitur, self.hasil_prepro, self.table_sudut_data = ekstraksi(
            file_citra)
        self.hasil_pre = Image.fromarray(self.hasil_prepro)
        self.hasil_pre = self.hasil_pre.resize((256, 256))
        self.hasil_pre = ImageTk.PhotoImage(self.hasil_pre)
        self.citra_glcm.configure(image=self.hasil_pre)
        self.citra_glcm.image = self.citra_glcm
        self.lbhKontras.configure(text=self.fitur[0])
        self.lbhHomog.configure(text=self.fitur[1])
        self.lbhEnergy.configure(text=self.fitur[2])
        self.lbhCor.configure(text=self.fitur[3])
        self.lbhEnt.configure(text=self.fitur[4])
        print("Hasil Ekstraksi Citra : ", self.fitur)
        print("=============================================")
        print("Table Sudut : ", self.table_sudut_data)
        print("=============================================")
        # Call the function to classify the Klasifikasi
        self.kelas = klasifikasiBP(self.fitur)
        if self.kelas == 0:
            print("Hasil : Mangga Belum Matang")
            self.hasil = 'Mangga Belum Matang'
        elif self.kelas == 1:
            print("Hasil : Mangga Matang")
            self.hasil = 'Mangga Matang'
        elif self.kelas == 2:
            print("Hasil : Mangga Setengah Matang")
            self.hasil = 'Mangga Setengah Matang'
        else:
            print("Hasil : Data Tidak Diketahui")
            self.hasil = 'Data Tidak Diketahui'
        self.hsBPNN.configure(text=self.hasil)

        # Populate the table_sudut data into the Treeview
        for row in self.table_sudut_data:
            self.table_sudut.insert("", "end", values=row)


root = tk.Tk()
my_gui = MyFirstGUI(root)
root.mainloop()
