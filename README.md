# Solutions_

I’ve updated pdf_checkbox_finder.py, but the error won’t go away until you install the Tesseract engine on your PC.

Fix

Download and install Tesseract for Windows from https://github.com/UB-Mannheim/tesseract/wiki. The default location is C:\Program Files\Tesseract-OCR\.
Run the script again with the updated file:
powershell
python pdf_checkbox_finder.py --pdf "C:/Girishbh/Pdf_extraction/Agb.pdf" --page 23 --question "C:/Girishbh/Pdf_extraction/cropped_question.png"

What I changed in the script

It now looks for tesseract.exe on your PATH and in the usual Windows install folders, so you shouldn’t need to edit the PATH.
If you installed it somewhere else, pass the location yourself:
powershell
  --tesseract-cmd "D:\Tools\Tesseract-OCR\tesseract.exe"
If Tesseract still isn’t found, the script now prints a short message with install instructions instead of a long traceback.
I replaced import fitz with import pymupdf, which removes the “fitz API is deprecated” warning. It falls back to fitz on older versions.
