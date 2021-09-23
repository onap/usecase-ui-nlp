#!/usr/bin/env python
# coding: utf-8

# auther = 'zhaoyehua'
# date = 20210918


from flask import Flask, request, render_template
import zipfile
import shutil
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'

@app.route("/uploader", methods=["POST"])
def uploader():
    print("aa")
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        return "ok"

@app.route("/unzipFile/<fileName>", methods=["GET"])
def unzipFile(fileName):
    zip_file = zipfile.ZipFile(os.path.join(app.config['UPLOAD_FOLDER'], fileName))
    parentDir = os.path.join(app.config['UPLOAD_FOLDER'], fileName.split(".zip")[0])
    if os.path.isdir(parentDir):
        pass
    else:
        os.mkdir(parentDir)
    for names in zip_file.namelist():
        zip_file.extract(names, parentDir)
    zip_file.close()
    return parentDir

@app.route("/deleteFile/<fileName>", methods=["GET"])
def deleteFile(fileName):
    path = os.path.join(app.config['UPLOAD_FOLDER'], fileName)
    if os.path.exists(path):
        os.remove(path)
        dirPath = path.split(".zip")[0]
        if os.path.isdir(dirPath):
            shutil.rmtree(dirPath)
            return "ok"
        return "ok"
    return "error: no such file"
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=33013)