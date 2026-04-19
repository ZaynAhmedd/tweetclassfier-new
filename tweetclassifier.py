import re
import pandas as pd
import streamlit as st

from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="X/Twitter Informative Post Classifier", layout="wide")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

informative_templates = [
    "Rescue teams are needed near the {place} after the {event}.",
    "Food packets and drinking water are urgently required at the {place} after the {event}.",
    "Heavy damage has been reported near the {place} because of the {event}.",
    "Medical help is needed for injured people near the {place} after the {event}.",
    "Evacuation has started in the {place} after the {event}.",
    "Power outage has affected the {place} after the {event}.",
    "Relief materials have been sent to the {place} after the {event}.",
    "Roads near the {place} are blocked after the {event}.",
    "People are stranded near the {place} because of the {event}.",
    "Emergency teams have reached the {place} after the {event}."
]

non_informative_templates = [
    "OMG this is insane near the {place}.",
    "I cannot believe what happened at the {place}.",
    "This is so sad man near the {place}.",
    "Prayers for everyone at the {place}.",
    "This is all over my timeline near the {place}.",
    "What a crazy day at the {place}.",
    "Everyone is talking about the {event} near the {place}.",
    "This feels unreal near the {place}.",
    "No words honestly about the {event} near the {place}.",
    "This is shocking near the {place}."
]

places = [
    "central bus stand","old city","district hospital","main bridge","railway station",
    "market area","coastal road","river bank","school ground","airport road",
    "highway junction","relief camp","industrial zone","town center","village road",
    "temple street","dam area","colony road","city outskirts","bus depot",
    "bridge road","sports ground","government school","main market","suburban area"
]

events = [
    "flood","cyclone","earthquake","landslide","fire",
    "storm","building collapse","bridge collapse","heavy rainfall","cloudburst"
]

@st.cache_data
def load_data():
    inf = []
    noninf = []

    for t,p,e in list(product(informative_templates,places,events))[:500]:
        inf.append((t.format(place=p,event=e),1))

    for t,p,e in list(product(non_informative_templates,places,events))[:500]:
        noninf.append((t.format(place=p,event=e),0))

    data = inf + noninf
    df = pd.DataFrame(data,columns=["text","target"])
    df = df.sample(frac=1,random_state=42).reset_index(drop=True)

    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len()>0].head(1000)

    return df

@st.cache_resource
def train():
    df = load_data()

    X = df["clean"]
    y = df["target"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    model = Pipeline([
        ("tfidf",TfidfVectorizer(max_features=8000,ngram_range=(1,2),stop_words="english")),
        ("clf",LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test,preds)
    report = classification_report(y_test,preds,output_dict=True)
    cm = confusion_matrix(y_test,preds)

    return model,acc,report,cm,df

def predict_one(model,text):
    t = clean_text(text)
    p = model.predict([t])[0]
    prob = model.predict_proba([t])[0]

    label = "Informative" if p==1 else "Non-Informative"
    return label,float(max(prob)*100)

def predict_csv(model,df):
    col = None
    for c in ["text","tweet_text","tweet","message","post"]:
        if c in df.columns:
            col = c
            break

    if col is None:
        raise ValueError("no text column")

    out = df.copy()
    out["clean"] = out[col].astype(str).apply(clean_text)
    out["pred"] = model.predict(out["clean"])
    probs = model.predict_proba(out["clean"])

    out["label"] = out["pred"].map({1:"Informative",0:"Non-Informative"})
    out["confidence"] = probs.max(axis=1)*100

    return out

st.markdown("## X/Twitter Informative Post Classifier")
st.caption("Zain Ahmed (22261A0541)")
st.write("This project tries to classify whether a tweet/post contains useful information (like updates, help requests, damage reports) or just general reactions/noise.")

model,acc,report,cm,df = train()

c1,c2,c3 = st.columns(3)
c1.metric("Rows",len(df))
c2.metric("Accuracy",round(acc,4))
c3.metric("Informative",int(df["target"].sum()))

tabs = st.tabs(["Single","CSV","Data"])

with tabs[0]:
    txt = st.text_area("Enter text")

    if st.button("Check"):
        if txt.strip():
            l,c = predict_one(model,txt)
            st.write("Prediction:",l)
            st.write("Confidence:",round(c,2),"%")

with tabs[1]:
    f = st.file_uploader("Upload CSV")

    if f is not None:
        try:
            d = pd.read_csv(f)
            r = predict_csv(model,d)
            st.dataframe(r.head(20))

            st.download_button("Download",r.to_csv(index=False),"output.csv")
        except Exception as e:
            st.error(e)

with tabs[2]:
    st.dataframe(df.head(20))
    st.dataframe(pd.DataFrame(cm))
    st.dataframe(pd.DataFrame(report).transpose())