# Gunakan Python versi 3.10
FROM python:3.10

# Atur direktori kerja di dalam container
WORKDIR /code

# Salin file requirements dan instal library
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Salin seluruh kode aplikasi
COPY . .

# Hugging Face Spaces WAJIB menggunakan port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]