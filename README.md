# Indonesia Smart License Plate Recognition

## Overview

Indonesia Smart License Plate Recognition adalah sistem Automatic License Plate Recognition (ALPR) berbasis Deep Learning yang dirancang untuk mendeteksi dan membaca plat nomor kendaraan Indonesia secara otomatis.

Sistem menggunakan model YOLOv8 untuk mendeteksi lokasi plat nomor dan EasyOCR untuk membaca karakter pada plat. Hasil OCR kemudian divalidasi menggunakan aturan format plat Indonesia dan dipetakan ke wilayah asal kendaraan.

## Features

* Deteksi plat nomor menggunakan YOLOv8
* OCR menggunakan EasyOCR
* Validasi format plat Indonesia
* Identifikasi wilayah kendaraan
* Confidence monitoring
* Dashboard interaktif berbasis Streamlit
* Upload gambar kendaraan
* Visualisasi hasil deteksi

## Technology Stack

* Python
* YOLOv8
* EasyOCR
* OpenCV
* Streamlit
* NumPy
* Pillow

## Workflow

Input Image

↓

YOLOv8 Detection

↓

Plate Cropping

↓

OCR Recognition

↓

Plate Validation

↓

Region Identification

↓

Result Dashboard

## Example Output

Plate Number : BA5746RD

Region : Sumatera Barat

YOLO Confidence : 86%

OCR Confidence : 99%

Status : VALID

## Project Structure

UAS_LicensePlate/

├── app.py

├── best.pt

├── requirements.txt

├── README.md

└── sample_images/

## Author

Reza Septian

Universitas Negeri Padang

Informatika