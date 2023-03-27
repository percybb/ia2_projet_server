from flask import Flask, request
import json
import os
import cv2
import pickle
import numpy as np
import werkzeug as wz
from descriptors import haralick_mean, bitdesc, glcm
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
# from sklearn.neighbors import neighbors_class

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)


classi = np.array(["1-Covid-Covid", "2-Covid-NomCovid", "3-CRCcancer-01_TUMOR", "4-CRCcancer-02_STROMA", "5-CRCcancer-03_COMPLEX", "6-CRCcancer-04_LYMPHO", "7-CRCcancer-05_DEBRIS", "8-CRCcancer-06_MUCOSA", "9-CRCcancer-07_ADIPOSE", "10-Glaucoma-Glaucoma", "11-Glaucoma-Saudavel", "12-KTHtips2-aluminium_foil", "13-KTHtips2-brown_bread", "14-KTHtips2-corduroy", "15-KTHtips2-cork", "16-KTHtips2-cotton", "17-KTHtips2-cracker", "18-KTHtips2-lettuce_leaf", "19-KTHtips2-linen",
                  "20-KTHtips2-white_bread", "21-KTHtips2-wood", "22-KTHtips2-wool", "23-outex24-1", "24-outex24-10", "25-outex24-11", "26-outex24-12", "27-outex24-13", "28-outex24-14", "29-outex24-15", "30-outex24-16", "31-outex24-17", "32-outex24-18", "33-outex24-19", "34-outex24-2", "35-outex24-20", "36-outex24-21", "37-outex24-22", "38-outex24-23", "39-outex24-24", "40-outex24-3", "41-outex24-4", "42-outex24-5", "43-outex24-6", "44-outex24-7", "45-outex24-8", "46-outex24-9"])

print(pickle.format_version)
models = []
#models.append(('LogReg', 'Logistic Regression'))
models.append(('DecTree', 'Decision Tree Classifier'))
#models.append(('RanFor', 'Random Forest'))
models.append(('KNearN', 'K-Nearest Neighbors'))
models.append(('SVC', 'Support Vector Classifier'))
models.append(('Cat', 'CatBoost'))

mods = []

transform = []
transform.append(('Normal', 'Normal'))
transform.append(('Res', 'Rescale'))
transform.append(('Stand', 'Standardization'))
#transform.append(('Norm', 'Normalization'))

transf = []
transf1 = []


for abr_mod, nom_mod in models:

    for abr_transfIn, nom_transfIn in transform:
        model_file = "./fichier/mod"+abr_mod+"_"+abr_transfIn+".pickle"
        local_classifier2 = pickle.load(open(model_file, "rb"))
        mods.append((nom_mod+nom_transfIn, local_classifier2))


for abr_transfIn, nom_transfIn in transform:

    if abr_transfIn != 'Normal':
        nomfile = './fichier/transf'+abr_transfIn+'.pickle'
        local_transf = pickle.load(open(nomfile, 'rb'))
        transf1.append((nom_transfIn, local_transf))


@app.route("/test", methods=(['GET']))
def mod0():
    return "hola "


@app.route("/upload", methods=['POST'])
def uploader():
    if request.method == 'POST':
        print("obtenemos dat")
        request_data = request.form
        #request_data = request.get_json(force=True)
        transform = request_data.get('transform')
        classifier = request_data.get('classifier')
        #transform = request_data.get['transform']
        #classifier = request_data.get['classifier']
        print(transform)
        print(classifier)

        #transform = "Normal"
        #classifier="Decision Tree Classifier"

        f = request.files['archivo']
        filename = f.filename
        # Guardamos el archivo en el directorio "Archivos PDF"
        f.save(os.path.join('upload', filename))
        # Retornamos una respuesta satisfactoria

        file_nom = "./upload/"+filename
        # print(fileName)
        img = cv2.imread(file_nom, 0)
        features = haralick_mean(img) + bitdesc(img) + glcm(img)
        # print(features)
        print(file_nom)
        new_data = np.array([features])
        # print(new_data)
        for a, b in mods:
            str1 = classifier+transform
            if a == str1:
                local_clas = b

        for c, d in transf1:
            if c == transform:
                local_opt = d

        print(transform)
        print(local_clas)
        predict = ""
        if transform == "Normal":
            predict = local_clas.predict(new_data)
            proba = local_clas.predict_proba(new_data)
        else:
            predict = local_clas.predict(local_opt.transform(new_data))
            proba = local_clas.predict_proba(local_opt.transform(new_data))

        print(predict)
        print(classi[int(predict)-1])
        print(proba[0][int(predict)-1])
        if classifier == "CatBoost":
            return {
                "predict": predict[0][0],
                "nomPredict": classi[int(predict)-1],
                "proba": proba[0][int(predict)-1]
            }
        else:
            return {
                "predict": predict[0],
                "nomPredict": classi[int(predict)-1],
                "proba": proba[0][int(predict)-1]
            }


if __name__ == '__main__':
    app.run(debug=True)

# app.run(port=8008, debug=True)
