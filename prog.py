import pandas as pd
import datetime
from datetime import date
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

data = pd.read_excel('mps.dataset.xlsx')

#Exclud sexul nul si modific aparitiile F si M in FEMININ si MASCULIN
data["sex"] = data["sex"].str.strip()
sex = data["sex"]
data = data[sex.notna()]
a = sex[sex == 'F']
sex.loc[sex.isin(a)] = "FEMININ"
a = sex[sex == 'M']
sex.loc[sex.isin(a)] = "MASCULIN"
data["sex"] = sex

#Exclud vasta nula
varsta = data["vârstă"]
data = data[varsta.notna()]

#Modific varsta de x ani si y luni -> x
a = varsta[varsta.str.contains('an|AN', na=False)]
d = a
d = d.str.split(expand=True, pat=" ")
varsta.loc[varsta.isin(a)] = d[0]

#Modific aparitiile de
#nou nascut / ora / zi / sapt / luna -> 0
a = varsta[varsta.str.contains('NOU NASCUT|nou nascut', na=False)]
varsta.loc[varsta.isin(a)] = "0"

a = varsta[varsta.str.contains('ore|ORE', na=False)]
varsta.loc[varsta.isin(a)] = "0"
a = varsta[varsta.str.contains('ora|ORA', na=False)]
varsta.loc[varsta.isin(a)] = "0"

a = varsta[varsta.str.contains('zi|ZI', na=False)]
varsta.loc[varsta.isin(a)] = "0"
a = varsta[varsta.str.contains('zile|ZILE', na=False)]
varsta.loc[varsta.isin(a)] = "0"

a = varsta[varsta.str.contains('sap|SAP', na=False)]
varsta.loc[varsta.isin(a)] = "0"

a = varsta[varsta.str.contains('luni|LUNI', na=False)]
varsta.loc[varsta.isin(a)] = "0"
a = varsta[varsta.str.contains('luna|LUNA', na=False)]
varsta.loc[varsta.isin(a)] = "0"

#Cast from text to number
a = pd.Series(range(0,101))
x = varsta[~varsta.isin(a)]
y = pd.to_numeric(x)
varsta.loc[varsta.isin(x)] = y

#Sterg varstele care sunt mai mari decat 100
a = pd.Series(range(0,101))
x = varsta[~varsta.isin(a)]
data = data[~(varsta.isin(x))]
data["vârstă"] = varsta

simptome_declarate = data["simptome declarate"]
#elimin coloanele nule (e mai mult pt verificare cand ma uit in excel ca
# nu stim ce eliminam deocamdata)
a = simptome_declarate[True == simptome_declarate.isnull()]
simptome_declarate.loc[simptome_declarate.isin(a)] = "nu"
data["simptome declarate"] = simptome_declarate.astype(str).str.lower()

#Pentru uniformizarea datelor pentru cei asimptomatici
regex_list = ['asim[a-z,\s,ă,-,+,0-9,.,=]*', '^[\s]*nu[\s]*[are]*','fara[\s,a-z]*','-','absente']

data["simptome declarate"] = data["simptome declarate"].replace(to_replace=regex_list, value="nu",regex=True)
#print(data['simptome declarate'])


# uniformizare mijloace de transport folosite
mijloace_de_transport_folosite = data["mijloace de transport folosite"]
# data = data[mijloace_de_transport_folosite.notna()]
a = mijloace_de_transport_folosite[True == mijloace_de_transport_folosite.isnull()]
mijloace_de_transport_folosite.loc[mijloace_de_transport_folosite.isin(a)] = "nu"
data["mijloace de transport folosite"] = mijloace_de_transport_folosite.astype(str).str.lower()

list_no = ['^n[a-z]*[\s,a-z]*', 'o', '0', 'fara[\s,a-z]*']
list_yes = ['masina[\s,a-z]*', '^a[a-z]*', '^d[a-z]*', 't[a-z]*', '1', 'da, nenu vezical']

data["mijloace de transport folosite"] = data["mijloace de transport folosite"].replace(to_replace=list_no, value="nu", regex=True)
data["mijloace de transport folosite"] = data["mijloace de transport folosite"].replace(to_replace=list_yes, value="da", regex=True)

# uniformizare contact cu o persoana infectata
contact_persoana_infectata = data["confirmare contact cu o persoană infectată"]
# data = data[contact_persoana_infectata.notna()]
a = contact_persoana_infectata[True == contact_persoana_infectata.isnull()]
contact_persoana_infectata.loc[contact_persoana_infectata.isin(a)] = "nu"
data["confirmare contact cu o persoană infectată"] = contact_persoana_infectata.astype(str).str.lower()

list_contact_negativ = ['0', '-', 'fara[\s,a-z]*', '^n[a-z]*[\s,a-z,ș]*', '^n[a-z]*', 'nuă']
list_contact_pozitiv = ['1', '^[d,v,p,c][\s,a-z,0-9,(,)]*', "focar familial"]

data["confirmare contact cu o persoană infectată"] = data["confirmare contact cu o persoană infectată"].replace(
                                                    to_replace=list_contact_negativ, value="nu", regex=True)

data["confirmare contact cu o persoană infectată"] = data["confirmare contact cu o persoană infectată"].replace(
                                                    to_replace=list_contact_pozitiv, value="da", regex=True)

#Modific coloana "istoric de calatorie"; 1-are istoric, 0-nu are istoric/relevanta
tara = data["istoric de călătorie"]
a = tara[tara == 'NU']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'NEAGĂ']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'NU ESTE CAZUL']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'NU A CALATORIT']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'NEAGA']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'NU E CAZUL']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'Nu']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'nu e cazul']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'nu']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == ' nu']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'nu ']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'NU A CALATORIT']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'Nu a calatorit']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'NU ARE']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'nu are']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'FARA']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'nu se stie']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'MU']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'NU ']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == ' NU']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'nu este cazul']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'Nu este cazul']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'nu  este cazul']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == ' NU A CALATORIT']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'NE ESTE CAZUL']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'NU A CALATORIT,']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'ASIMPTOMATIC']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'DURERE GAMBA STG,IMPOTENTA FUNCTIONALA GAMBA STG']
tara.loc[tara.isin(a)] = "0"

a = tara[tara == 'SCOTIA-ROMANIA']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'da 02.02.2020-30.04.2020']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'Germania06-.05.-08.05.2020']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'DA']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'Tata -sofer de tir (Franta 02.05-07.05.2020)']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'DA; 10-13.05.2020']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'GERMANIA, 17.01.2020-04.05.2020']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'PORTUGALIA IAN 2020']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'INTERNATA IN SPITALUL JUDETEA "SF.IOAN CEL NOU"SUCEAVA 18-23.03.2020']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'da-sotia SUA']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'VENITA DIN ANGLIA IN DATA DE 18.05.2020 CU MASINA PERSONALA']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'DA;12.05.2020']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'GERMANIA']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'Austria']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'da-Suedia, intors in 1 martie']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == ' DA 17.05.2020']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == '08-10.04.2020 SCOTIA,ROMANIA']
tara.loc[tara.isin(a)] = "1"

a = tara[tara == 'DA ']
tara.loc[tara.isin(a)] = "1"

a = tara[True == tara.isnull()]
tara.loc[tara.isin(a)] = "0"

data["istoric de călătorie"] = tara


#print (data['simptome declarate'])
data.to_excel("output_dataset.xlsx")

import re

substring_to_ignore = ['1','2','3','4','5','6','7','8','9','0']

for entrance in data["simptome declarate"]:
  if (type(entrance) !=list):
    index = data[data["simptome declarate"]==entrance].index.values.astype(int)
    for idx in index:
      data["simptome declarate"][idx]=  "".join(data["simptome declarate"][idx].split())
      data["simptome declarate"][idx] = re.split(",", entrance)
      data["simptome declarate"][idx] = [x.strip(' ') for x in data["simptome declarate"][idx]]

      for simptom in data["simptome declarate"][idx]:
        word_idx = data["simptome declarate"][idx].index(simptom)

        if 'dureri abdominale' in simptom or 'durere a' in simptom or 'dureri a' in simptom:
           data["simptome declarate"][idx][word_idx]= 'durere abdominala'
        if 'dureri abdominale' in simptom:
           data["simptome declarate"][idx][word_idx]= 'durere abdominala'
        if 'tuse' in simptom or ' tuse' in simptom:
          data["simptome declarate"][idx][word_idx]= 'tuse'
        if 'febr' in simptom or 'temp' in simptom:
          data["simptome declarate"][idx][word_idx]= 'febra'
        if 'disp' in simptom  or 'dips' in simptom or 'dipn' in simptom:
          data["simptome declarate"][idx][word_idx]= 'dispnee'
        if 'friso' in simptom:
          data["simptome declarate"][idx][word_idx]= 'frisoane'
        if 'edem' in simptom:
          data["simptome declarate"][idx][word_idx]= 'edeme'
        if 'cefale' in simptom:
          data["simptome declarate"][idx][word_idx]= 'cefalee'
        for ignore in substring_to_ignore:
          if ignore in simptom:
            if simptom in  data["simptome declarate"][idx]:
              data["simptome declarate"][idx].remove( data["simptome declarate"][idx][word_idx])
mlb = MultiLabelBinarizer()

df = pd.DataFrame(mlb.fit_transform(data["simptome declarate"]),columns=mlb.classes_, index=data.index)

data.sex.replace(to_replace=dict(FEMININ=0, MASCULIN=1, masculin=1), inplace=True)
data['mijloace de transport folosite'].replace(to_replace=dict(nu=0, da=1), inplace=True)
data["confirmare contact cu o persoană infectată"] .replace(to_replace=dict(nu=0, da=1), inplace=True)

encoded = pd.concat([data, df], axis = 1)
encoded = encoded.drop(["rezultat testare"],axis = 1)
encoded = encoded.drop(["dată debut simptome declarate"],axis = 1)
encoded = encoded.drop(["dată internare"],axis = 1)
encoded = encoded.drop(["data rezultat testare"],axis = 1)
encoded = encoded.drop(["simptome raportate la internare"],axis = 1)
encoded = encoded.drop(["simptome declarate"],axis = 1)

# data["rezultat testare"]
encoded = pd.concat([encoded, data["rezultat testare"]], axis =1)
encoded['rezultat testare'].replace(to_replace=dict(NEGATIV=0, POZITIV=1, NECONCLUDENT=0, NEGATIB=0), inplace=True)
encoded.dropna(subset=['rezultat testare'], inplace=True)
encoded = encoded.drop(["instituția sursă"], axis = 1)
encoded = encoded.drop(["diagnostic și semne de internare"], axis = 1)
encoded = encoded.drop([""], axis = 1)

encoded.to_excel("encoded.xlsx")
y_data = encoded['rezultat testare']

x_data = encoded.drop('rezultat testare', axis = 1)
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.2)
print(y_test_data)

model = LogisticRegression()
model.fit(x_training_data, y_training_data)

predictions1 = model.predict(x_test_data)
print(classification_report(y_test_data, predictions1))

predictions2 = model.predict(x_training_data)
print(classification_report(y_training_data, predictions2))

print('Accuracy: '+ str(accuracy_score(y_test_data, predictions1)))
print('F1 score: '+ str(f1_score(y_test_data, predictions1,average='weighted')))
print('Recall: ' + str(recall_score(y_test_data, predictions1, average='weighted')))
print('Precision: ' + str(precision_score(y_test_data, predictions1, average='weighted')))
print('\n clasification report:\n' + str(classification_report(y_test_data, predictions1)))
print('\n confussion matrix:\n' + str(confusion_matrix(y_test_data, predictions1)))
