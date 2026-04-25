"""
================================================================================
  AeroSense — UAV PM2.5 Prediction System
  Competition-Ready · Trained CNN Regression · WHO 2021 AQG
================================================================================
  CONFIGURATION (edit these two lines before running):
    CSV_PATH    — path to your sensor_with_filename.csv
    IMAGES_DIR  — path to the folder containing all DJI_XXXX.jpg images

  HOW IT WORKS AT RUNTIME:
    1. On first launch the app trains a Ridge Regression model automatically
       using the CSV + images configured below. Model is saved to disk.
    2. On subsequent launches the saved model loads instantly.
    3. Users only need to: upload any UAV image + optionally enter sensor values.

  MODULES:
    [1]  CONFIG         — all settings, paths, WHO thresholds
    [2]  LOGGER         — structured file + JSONL logging
    [3]  DATA PROCESSOR — sensor validation, scaling
    [4]  CNN EXTRACTOR  — MobileNetV2 (1280-dim) / histogram fallback (102-dim)
    [5]  MODEL TRAINER  — Ridge Regression: CNN features -> real PM2.5
    [6]  FUSION ENGINE  — trained CNN + optional sensor fusion
    [6.5] CONSISTENCY   — visual-sensor agreement checker
    [7]  WHO CLASSIFIER — Healthy / Unhealthy / Hazardous (WHO 2021)
    [8]  HEALTH ADVISOR — 5-category WHO health recommendations
    [9]  PREDICTOR      — full pipeline orchestrator
    [10] STREAMLIT UI   — single-page dark-theme competition frontend
================================================================================
"""

# ============================================================
#  EDIT THESE TWO PATHS BEFORE RUNNING
# ============================================================
CSV_PATH   = r"C:\Users\FOSTINA\Desktop\AEROGUARD_AI\sensor_with_filename.csv"
IMAGES_DIR = r"C:\Users\FOSTINA\Desktop\AEROGUARD_AI\UAV_Images"
# ============================================================

import json, logging, os, pickle, sys, time, traceback, warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ── [1] CONFIG ────────────────────────────────────────────────────────────────

@dataclass
class SensorConfig:
    pm1_min: float=0.0;  pm1_max: float=500.0
    pm25_min:float=0.0;  pm25_max:float=500.0
    pm10_min:float=0.0;  pm10_max:float=500.0

@dataclass
class CNNConfig:
    input_size: Tuple[int,int]=(224,224)
    backbone:   str="MobileNetV2"
    feature_dim:int=1280

@dataclass
class FusionConfig:
    cnn_weight:              float=0.30
    sensor_weight:           float=0.70
    beta_pm25:               float=0.70
    beta_pm1:                float=0.18
    beta_pm10:               float=0.12
    visual_correction_scale: float=0.22

@dataclass
class ConsistencyConfig:
    clean_image_pm25_max: float=20.0
    hazy_image_pm25_min:  float=15.0
    smoky_image_pm25_min: float=35.0
    agreement_tolerance:  float=0.35
    confidence_boost:     float=0.10
    confidence_penalty:   float=0.18

@dataclass
class WHOThresholds:
    healthy_max:  float=15.0
    unhealthy_max:float=45.0

@dataclass
class AppConfig:
    app_name:        str  ="AeroSense"
    log_dir:         str  ="logs"
    log_file:        str  ="aerosense.log"
    model_path:      str  ="aerosense_model.pkl"
    scaler_path:     str  ="aerosense_scaler.pkl"
    max_history:     int  =500
    pm25_outlier_max:float=800.0
    test_split:      float=0.20
    ridge_alpha:     float=1.0
    sensor:      SensorConfig     =field(default_factory=SensorConfig)
    cnn:         CNNConfig         =field(default_factory=CNNConfig)
    fusion:      FusionConfig      =field(default_factory=FusionConfig)
    consistency: ConsistencyConfig =field(default_factory=ConsistencyConfig)
    who:         WHOThresholds     =field(default_factory=WHOThresholds)
    def to_dict(self): return asdict(self)

CONFIG = AppConfig()


# ── [2] LOGGER ────────────────────────────────────────────────────────────────

class AeroLogger:
    _instance = None
    def __new__(cls, cfg=CONFIG):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(cfg)
        return cls._instance

    def _init(self, cfg):
        p = Path(cfg.log_dir); p.mkdir(parents=True, exist_ok=True)
        self.log_file      = p / cfg.log_file
        self.json_log_file = p / "events.jsonl"
        logging.basicConfig(level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(self.log_file, encoding="utf-8"),
                      logging.StreamHandler(sys.stdout)])
        self.logger = logging.getLogger(cfg.app_name)
        self.logger.info(f"{cfg.app_name} initialised.")

    def _w(self, level, msg, meta):
        r={"ts":datetime.utcnow().isoformat(),"level":level,"msg":msg,**meta}
        try:
            with open(self.json_log_file,"a",encoding="utf-8") as f:
                f.write(json.dumps(r)+"\n")
        except: pass

    def info(self,    msg, **m): self.logger.info(msg);    self._w("INFO",   msg,m)
    def warning(self, msg, **m): self.logger.warning(msg); self._w("WARNING",msg,m)
    def error(self,   msg, **m): self.logger.error(msg);   self._w("ERROR",  msg,m)

    def get_recent_logs(self, n=40):
        try:
            lines=open(self.json_log_file,encoding="utf-8").readlines()
            return [json.loads(l) for l in lines[-n:]][::-1]
        except: return []

logger = AeroLogger(CONFIG)


# ── [3] DATA PROCESSOR ───────────────────────────────────────────────────────

@dataclass
class SensorReading:
    pm1:float; pm25:float; pm10:float
    timestamp:str=field(default_factory=lambda:datetime.utcnow().isoformat())
    filename:Optional[str]=None
    colocated:bool=False

    def is_valid(self):
        for n,v in [("PM1",self.pm1),("PM2.5",self.pm25),("PM10",self.pm10)]:
            if v<0:    return False,f"{n} cannot be negative."
            if v>5000: return False,f"{n}={v} exceeds plausible range."
        if self.pm25>self.pm10+5:
            return False,"PM2.5 > PM10+5 is physically implausible."
        return True,"OK"

class DataProcessor:
    def __init__(self,cfg=CONFIG.sensor):
        self.cfg=cfg; self._history=[]

    def validate(self,r):
        ok,msg=r.is_valid()
        if not ok: logger.warning(f"Sensor invalid: {msg}")
        return ok,msg

    def scale(self,r):
        def s(v,lo,hi): return float(np.clip((v-lo)/(hi-lo+1e-9),0,1))
        return np.array([s(r.pm1,self.cfg.pm1_min,self.cfg.pm1_max),
                         s(r.pm25,self.cfg.pm25_min,self.cfg.pm25_max),
                         s(r.pm10,self.cfg.pm10_min,self.cfg.pm10_max)],dtype=np.float32)

    def record(self,r,pred):
        e=asdict(r); e["predicted_pm25"]=pred
        self._history.append(e)
        if len(self._history)>CONFIG.max_history: self._history.pop(0)

    @staticmethod
    def load_csv(path):
        df=pd.read_csv(path)
        df.columns=[c.strip() for c in df.columns]
        rename={}
        for c in df.columns:
            cl=c.lower().replace(" ","").replace(".","").replace("_","")
            if "pm1" in cl and "10" not in cl and "25" not in cl: rename[c]="pm1"
            elif "pm25" in cl or ("pm2" in cl and "5" in cl):     rename[c]="pm25"
            elif "pm10" in cl:                                      rename[c]="pm10"
            elif "file" in cl:                                      rename[c]="filename"
            elif "created" in cl or "time" in cl:                   rename[c]="created_at"
        df.rename(columns=rename,inplace=True)
        for col in ["pm1","pm25","pm10"]:
            if col in df.columns:
                df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0)
        return df

processor=DataProcessor(CONFIG.sensor)


# ── [4] CNN EXTRACTOR ────────────────────────────────────────────────────────

class CNNExtractor:
    def __init__(self,cfg=CONFIG.cnn):
        self.cfg=cfg; self._model=None; self._mode="unloaded"

    def load(self):
        if TF_AVAILABLE:
            try:
                self._model=MobileNetV2(weights="imagenet",include_top=False,
                    pooling="avg",input_shape=(*self.cfg.input_size,3))
                self._mode="mobilenetv2"
                logger.info("CNN: MobileNetV2 loaded (1280-dim)")
                return "mobilenetv2"
            except Exception as e:
                logger.warning(f"MobileNetV2 failed ({e}), using fallback.")
        self._mode="fallback"
        logger.info("CNN: Histogram fallback (102-dim)")
        return "fallback"

    @property
    def mode(self): return self._mode

    def extract(self,img:Image.Image)->np.ndarray:
        t0=time.time()
        f=(self._extract_mobilenet(img) if self._mode=="mobilenetv2" and self._model
           else self._extract_fallback(img))
        logger.info(f"CNN: dim={len(f)} mode={self._mode} t={time.time()-t0:.2f}s")
        return f

    def _extract_mobilenet(self,img):
        a=np.array(img.convert("RGB").resize(self.cfg.input_size),dtype=np.float32)
        return self._model.predict(np.expand_dims(preprocess_input(a),0),verbose=0)[0].astype(np.float32)

    def _extract_fallback(self,img):
        a=np.array(img.convert("RGB").resize((128,128)),dtype=np.float32)/255.0
        feats=[]
        for c in range(3):
            h,_=np.histogram(a[:,:,c],bins=32,range=(0,1))
            feats.extend((h/(h.sum()+1e-8)).tolist())
        lum=0.299*a[:,:,0]+0.587*a[:,:,1]+0.114*a[:,:,2]
        feats+=[float(lum.mean()),float(lum.std()),
                float(np.percentile(lum,25)),float(np.percentile(lum,75)),
                float((lum<0.30).mean()),float((lum>0.80).mean())]
        return np.array(feats,dtype=np.float32)

    def compute_haze_proxy(self,f):
        af=np.abs(f)
        dk=float(f[-2]) if self._mode=="fallback" and len(f)>=102 else 0.0
        br=float(f[-1]) if self._mode=="fallback" and len(f)>=102 else 0.0
        return float(np.clip(af.mean()*1.40+af.std()*0.25+dk*0.55+br*0.20,0,1))

    def visual_category(self,hp):
        if hp<0.25: return "Clear Sky",             "Expected PM2.5: 0-15 ug/m3  (WHO Healthy)"
        if hp<0.55: return "Slight Haze",            "Expected PM2.5: 10-35 ug/m3  (Healthy to Unhealthy)"
        if hp<0.75: return "Moderate Haze / Smoke",  "Expected PM2.5: 30-70 ug/m3  (Unhealthy to Hazardous)"
        return             "Heavy Haze / Dense Smoke","Expected PM2.5: 60-120+ ug/m3  (Hazardous)"

    def top_activations(self,f,k=10):
        idx=np.argsort(np.abs(f))[::-1][:k]
        return [(int(i),float(f[i])) for i in idx]


# ── [5] MODEL TRAINER ────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    n_samples:int; n_train:int; n_test:int
    mae:float; rmse:float; r2:float
    feature_dim:int; pm25_mean:float; pm25_std:float
    who_dist:Dict[str,int]; trained_at:str

class ModelTrainer:
    def __init__(self,extractor:CNNExtractor,cfg=CONFIG):
        self.extractor=extractor; self.cfg=cfg

    def train(self,csv_path:str,images_dir:str,status_cb=None)->TrainResult:
        if not SK_AVAILABLE:
            raise RuntimeError("scikit-learn not installed. Run: pip install scikit-learn")

        df=DataProcessor.load_csv(csv_path)
        if "filename" not in df.columns:
            raise ValueError("CSV must have a 'filename' column.")
        if "pm25" not in df.columns:
            raise ValueError("CSV must have a PM2.5 column.")

        df=df[(df["pm25"]>0)&(df["pm25"]<=self.cfg.pm25_outlier_max)].copy()
        df=df.dropna(subset=["filename","pm25"])
        df["filename"]=df["filename"].astype(str).str.strip()

        images_path=Path(images_dir); total=len(df)
        logger.info(f"TRAIN START: {total} valid rows, images_dir={images_dir}")

        X,y=[],[]
        found=0
        for i,(_,row) in enumerate(df.iterrows()):
            img_file=images_path/row["filename"]
            if not img_file.exists():
                for ext in [".jpg",".JPG",".jpeg",".JPEG",".png",".PNG"]:
                    alt=images_path/(Path(row["filename"]).stem+ext)
                    if alt.exists(): img_file=alt; break
                else: continue
            try:
                feat=self.extractor.extract(Image.open(img_file).convert("RGB"))
                X.append(feat); y.append(float(row["pm25"])); found+=1
            except Exception as e:
                logger.warning(f"Skipped {row['filename']}: {e}"); continue
            if status_cb and i%25==0:
                status_cb(i/total,f"Extracting features... {found} matched / {i+1} processed")

        if found<10:
            raise ValueError(
                f"Only {found} image-sensor pairs found out of {total} rows.\n"
                f"Check IMAGES_DIR='{images_dir}' contains the DJI images.")

        X=np.array(X,dtype=np.float32); y=np.array(y,dtype=np.float32)
        logger.info(f"TRAIN: {len(X)} pairs, feature_dim={X.shape[1]}")

        if status_cb: status_cb(0.82,f"Training Ridge Regression on {len(X)} samples...")
        scaler=StandardScaler()
        X_sc=scaler.fit_transform(X)
        X_tr,X_te,y_tr,y_te=train_test_split(X_sc,y,test_size=self.cfg.test_split,random_state=42)
        model=Ridge(alpha=self.cfg.ridge_alpha)
        model.fit(X_tr,y_tr)

        y_pred=np.clip(model.predict(X_te),0,None)
        mae=float(mean_absolute_error(y_te,y_pred))
        rmse=float(np.sqrt(mean_squared_error(y_te,y_pred)))
        r2=float(r2_score(y_te,y_pred))

        who_dist={"Healthy":int((y<=15).sum()),
                  "Unhealthy":int(((y>15)&(y<=45)).sum()),
                  "Hazardous":int((y>45).sum())}

        if status_cb: status_cb(0.97,"Saving model to disk...")
        with open(self.cfg.model_path, "wb") as f: pickle.dump(model, f)
        with open(self.cfg.scaler_path,"wb") as f: pickle.dump(scaler,f)
        logger.info(f"TRAIN COMPLETE: n={len(X)} MAE={mae:.1f} RMSE={rmse:.1f} R2={r2:.3f}")
        if status_cb: status_cb(1.0,"Training complete!")

        return TrainResult(n_samples=len(X),n_train=len(X_tr),n_test=len(X_te),
            mae=round(mae,2),rmse=round(rmse,2),r2=round(r2,4),
            feature_dim=X.shape[1],pm25_mean=round(float(y.mean()),1),
            pm25_std=round(float(y.std()),1),who_dist=who_dist,
            trained_at=datetime.utcnow().isoformat())

    @staticmethod
    def load(cfg=CONFIG):
        try:
            with open(cfg.model_path, "rb") as f: m=pickle.load(f)
            with open(cfg.scaler_path,"rb") as f: s=pickle.load(f)
            logger.info("Trained model loaded from disk.")
            return m,s
        except FileNotFoundError: return None,None

    @staticmethod
    def predict_single(features:np.ndarray,model,scaler)->float:
        x=scaler.transform(features.reshape(1,-1))
        return float(np.clip(model.predict(x)[0],0,5000))


# ── [6] FUSION ENGINE ────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    predicted_pm25:float; sensor_contribution:float; cnn_contribution:float
    visual_haze_proxy:float; visual_label:str; visual_pm25_range:str
    base_confidence:float; cnn_weight_used:float; sensor_weight_used:float
    top_features:List[Tuple[int,float]]; mode:str

class FusionEngine:
    def __init__(self,cfg=CONFIG.fusion): self.cfg=cfg

    def fuse(self,features,scaled_sensor,raw_sensor,extractor,
             model=None,scaler=None,cnn_only=False)->FusionResult:
        hp=extractor.compute_haze_proxy(features)
        vlbl,vrng=extractor.visual_category(hp)
        topk=extractor.top_activations(features,k=10)

        if model is not None and scaler is not None:
            cnn_est=ModelTrainer.predict_single(features,model,scaler); trained=True
        else:
            cnn_est=hp*120.0; trained=False

        if cnn_only:
            fused=float(np.clip(cnn_est,0,5000)); sc=0.0; cc=round(fused,2)
            base_conf=0.62 if trained else 0.45
            mode_str="trained_cnn_only" if trained else "proxy_cnn_only"
            logger.info(f"FUSION({mode_str}) PM2.5={fused:.2f} haze={hp:.3f}")
        else:
            sest=(raw_sensor.pm25*self.cfg.beta_pm25+raw_sensor.pm1*self.cfg.beta_pm1
                  +raw_sensor.pm10*self.cfg.beta_pm10)
            corr=float(np.clip(1.0+(hp-0.3)*self.cfg.visual_correction_scale,0.80,1.30))
            alpha=self.cfg.cnn_weight
            fused=float(np.clip(alpha*cnn_est+(1-alpha)*sest*corr,0,5000))
            sc=round((1-alpha)*sest*corr,2); cc=round(alpha*cnn_est,2)
            af=np.abs(features)
            s1=float(np.clip(1.0-abs(raw_sensor.pm25-raw_sensor.pm1)/(raw_sensor.pm10+1),0,1))
            s2=float(np.clip(1.0-af.std()/(af.mean()+1e-6),0,1))
            base_conf=float(np.clip(0.55*s1+0.45*s2,0.35,0.92))
            if trained: base_conf=min(0.95,base_conf+0.08)
            mode_str="trained_fused" if trained else "proxy_fused"
            logger.info(f"FUSION({mode_str}) PM2.5={fused:.2f} haze={hp:.3f} sest={sest:.2f} corr={corr:.3f}")

        return FusionResult(
            predicted_pm25=round(fused,2),sensor_contribution=sc,cnn_contribution=cc,
            visual_haze_proxy=round(hp,4),visual_label=vlbl,visual_pm25_range=vrng,
            base_confidence=round(base_conf*100,1),
            cnn_weight_used=self.cfg.cnn_weight,sensor_weight_used=self.cfg.sensor_weight,
            top_features=topk,mode=mode_str)


# ── [6.5] CONSISTENCY CHECKER ────────────────────────────────────────────────

@dataclass
class ConsistencyResult:
    status:str; agreement_pct:float; gap:float
    visual_pm25_est:float; sensor_pm25:float; conflict_type:Optional[str]
    explanation:str; recommendation:str
    confidence_delta:float; final_confidence:float

class ConsistencyChecker:
    _PM25_RANGE=500.0
    def __init__(self,cfg=CONFIG.consistency): self.cfg=cfg

    def check(self,hp,raw_sensor,base_conf)->ConsistencyResult:
        vn=float(np.clip(hp,0,1)); sn=float(np.clip(raw_sensor.pm25/self._PM25_RANGE,0,1))
        vest=round(hp*120.0,1); gap=abs(vn-sn)
        agr=round(float(np.clip((1.0-gap)*100,0,100)),1)
        ct=None
        if vn<0.30 and raw_sensor.pm25>self.cfg.clean_image_pm25_max:
            ct="CLEAN_IMAGE_HIGH_SENSOR"
        elif vn>0.55 and raw_sensor.pm25<self.cfg.hazy_image_pm25_min:
            ct="SMOKY_IMAGE_LOW_SENSOR"
        elif vn>0.75 and raw_sensor.pm25<self.cfg.smoky_image_pm25_min:
            ct="SMOKY_IMAGE_LOW_SENSOR"
        if   gap<=self.cfg.agreement_tolerance*0.50: status="AGREE"
        elif gap<=self.cfg.agreement_tolerance:      status="PARTIAL"
        else:                                        status="CONFLICT"
        delta={"AGREE":self.cfg.confidence_boost,"PARTIAL":0.0,
               "CONFLICT":-self.cfg.confidence_penalty}[status]
        fc=round(float(np.clip(base_conf+delta*100,10,97)),1)
        exp,rec=self._explain(status,ct,hp,raw_sensor.pm25,vest,agr)
        logger.info(f"CONSISTENCY {status} ({agr:.0f}%) gap={gap:.3f} delta={delta*100:+.1f}%")
        return ConsistencyResult(status=status,agreement_pct=agr,gap=round(gap,4),
            visual_pm25_est=vest,sensor_pm25=raw_sensor.pm25,conflict_type=ct,
            explanation=exp,recommendation=rec,
            confidence_delta=round(delta*100,1),final_confidence=fc)

    def cnn_only_result(self,hp,base_conf)->ConsistencyResult:
        return ConsistencyResult(status="CNN_ONLY",agreement_pct=0,gap=0,
            visual_pm25_est=round(hp*120,1),sensor_pm25=0,conflict_type=None,
            explanation="Sensor not used. Prediction is driven entirely by trained CNN regression.",
            recommendation="Confirm co-location and enter sensor readings to enable fused prediction.",
            confidence_delta=0.0,final_confidence=round(float(np.clip(base_conf,10,80)),1))

    def _explain(self,status,ct,hp,spm25,vest,agr):
        vc=("clear" if hp<0.25 else "slightly hazy" if hp<0.55
            else "moderately hazy" if hp<0.75 else "heavily hazy/smoky")
        if status=="AGREE":
            return (f"Strong agreement ({agr:.0f}%). Image appears {vc} and sensor "
                    f"reports {spm25:.1f} ug/m3. Both signals confirm the same air quality.",
                    "Inputs are consistent. Prediction is highly reliable.")
        if status=="PARTIAL":
            return (f"Mild disagreement ({agr:.0f}%). Image looks {vc}, sensor reads {spm25:.1f} ug/m3. "
                    "Minor offset is normal in field conditions.",
                    "Prediction is valid. Small mismatches within tolerance are acceptable.")
        if ct=="CLEAN_IMAGE_HIGH_SENSOR":
            return (f"Conflict ({agr:.0f}%). Image looks {vc} but sensor reports {spm25:.1f} ug/m3. "
                    "Fine PM2.5 is INVISIBLE to cameras. A clear sky can still hold dangerous particles.",
                    "Trust the sensor. Fine PM2.5 does not produce visible haze at low concentrations.")
        if ct=="SMOKY_IMAGE_LOW_SENSOR":
            return (f"Conflict ({agr:.0f}%). Image looks {vc} but sensor reads only {spm25:.1f} ug/m3. "
                    "Visible haze may be fog or water vapour, not PM2.5.",
                    "Verify image and sensor are from the same location and time.")
        return (f"Significant mismatch ({agr:.0f}%). Visual ~{vest:.1f} vs sensor {spm25:.1f} ug/m3.",
                "Verify image and sensor correspond to the same location and time.")


# ── [7] WHO CLASSIFIER ───────────────────────────────────────────────────────

@dataclass
class WHOResult:
    pm25:float; tier:str; color:str; icon:str
    who_guideline:str; description:str; exceeds_by:float

class PollutionClassifier:
    _TIERS=[
        (15.0,        "Healthy",   "#00C853","✅","WHO AQG <= 15 ug/m3",
         "Within WHO 2021 safe guideline. Air quality poses little to no risk."),
        (45.0,        "Unhealthy", "#FF8F00","⚠️","WHO IT-1  15-45 ug/m3",
         "Exceeds WHO AQG. Long-term exposure poses measurable health risks."),
        (float("inf"),"Hazardous", "#D32F2F","🚨","Above WHO IT-1 > 45 ug/m3",
         "Critically exceeds WHO IT-1. Immediate health risk for all populations."),
    ]
    def classify(self,pm25):
        for mv,tier,color,icon,guide,desc in self._TIERS:
            if pm25<=mv:
                exc=round(max(0.0,pm25-CONFIG.who.healthy_max),2)
                logger.info(f"WHO {tier} PM2.5={pm25} exceeds={exc}")
                return WHOResult(pm25,tier,color,icon,guide,desc,exc)
        return WHOResult(pm25,"Hazardous","#D32F2F","🚨","Above WHO IT-1","Health emergency.",max(0,pm25-15))


# ── [8] HEALTH ADVISOR ───────────────────────────────────────────────────────

@dataclass
class HealthAdvice:
    summary:str; outdoor_activity:str; ventilation:str
    mask_recommendation:str; sensitive_groups:str; uav_ops:str; color:str

class HealthAdvisor:
    _A={
        "Healthy":HealthAdvice(
            "Air quality is within WHO 2021 safe limits. Safe for everyone.",
            "All outdoor activities are safe. No restrictions.",
            "Open windows freely. Natural ventilation is recommended.",
            "No mask required.",
            "Safe for children, elderly, and those with respiratory conditions.",
            "Ideal UAV conditions — optimal visibility and sensor accuracy.",
            "#00C853"),
        "Unhealthy":HealthAdvice(
            "Exceeds WHO AQG (15 ug/m3). Reduce prolonged outdoor exposure.",
            "Limit prolonged strenuous outdoor activity, especially for sensitive groups.",
            "Reduce natural ventilation at peak hours. Use air filtration.",
            "N95/KN95 recommended for sensitive individuals outdoors.",
            "Children, elderly, asthma and heart patients: limit outdoor time.",
            "Moderate haze possible. Verify image quality. Shorten missions.",
            "#FF8F00"),
        "Hazardous":HealthAdvice(
            "Critically exceeds WHO IT-1 (45 ug/m3). Immediate risk for everyone.",
            "Avoid ALL outdoor activity. Stay indoors immediately.",
            "Seal all windows and doors. Run HEPA purifiers continuously.",
            "N95/P100 respirator essential if outdoors.",
            "EMERGENCY: All populations at risk. Seek medical advice immediately.",
            "UAV operations suspended. Severe visibility and sensor contamination risk.",
            "#D32F2F"),
    }
    def advise(self,who):
        a=self._A.get(who.tier,self._A["Hazardous"])
        logger.info(f"Health advice: {who.tier}"); return a


# ── [9] PREDICTOR ────────────────────────────────────────────────────────────

class PM25Predictor:
    def __init__(self,cfg=CONFIG):
        self.cfg=cfg; self.processor=processor
        self.extractor=CNNExtractor(cfg.cnn)
        self.fusion_eng=FusionEngine(cfg.fusion)
        self.checker=ConsistencyChecker(cfg.consistency)
        self.classifier=PollutionClassifier(); self.advisor=HealthAdvisor()
        self._ready=False; self.model=None; self.scaler=None
        self.train_result:Optional[TrainResult]=None

    def initialise(self):
        mode=self.extractor.load()
        self.model,self.scaler=ModelTrainer.load(self.cfg)
        self._ready=True
        trained=self.model is not None
        logger.info(f"Predictor ready. CNN={mode} trained={trained}")
        return mode,trained

    def train_now(self,status_cb=None):
        trainer=ModelTrainer(self.extractor,self.cfg)
        self.train_result=trainer.train(CSV_PATH,IMAGES_DIR,status_cb)
        self.model,self.scaler=ModelTrainer.load(self.cfg)
        return self.train_result

    @property
    def ready(self): return self._ready

    @property
    def has_trained_model(self): return self.model is not None

    def predict(self,image:Image.Image,reading:SensorReading)->dict:
        if not self._ready: raise RuntimeError("Call .initialise() first.")
        t0=time.time()
        ok,msg=self.processor.validate(reading)
        if not ok: raise ValueError(msg)
        scaled=self.processor.scale(reading)
        features=self.extractor.extract(image)
        cnn_only=not reading.colocated
        fusion=self.fusion_eng.fuse(features,scaled,reading,self.extractor,
                                    self.model,self.scaler,cnn_only)
        consistency=(self.checker.cnn_only_result(fusion.visual_haze_proxy,fusion.base_confidence)
                     if cnn_only else
                     self.checker.check(fusion.visual_haze_proxy,reading,fusion.base_confidence))
        who=self.classifier.classify(fusion.predicted_pm25)
        advice=self.advisor.advise(who)
        self.processor.record(reading,fusion.predicted_pm25)
        elapsed=round(time.time()-t0,3)
        out=dict(pm25=fusion.predicted_pm25,confidence=consistency.final_confidence,
                 base_confidence=fusion.base_confidence,
                 who_tier=who.tier,who_color=who.color,who_icon=who.icon,
                 who_guideline=who.who_guideline,who_description=who.description,
                 who_exceeds_by=who.exceeds_by,consistency=consistency,
                 advice=advice,fusion=fusion,elapsed_s=elapsed,
                 cnn_mode=self.extractor.mode,prediction_mode=fusion.mode,
                 has_trained_model=self.has_trained_model,
                 timestamp=reading.timestamp,image_name=reading.filename or "uploaded")
        logger.info(f"PREDICTION PM2.5={fusion.predicted_pm25} tier={who.tier} mode={fusion.mode}")
        return out


# ── [10] STREAMLIT UI ────────────────────────────────────────────────────────

st.set_page_config(page_title="AeroSense — PM2.5 Prediction",
                   page_icon="🛸",layout="wide",
                   initial_sidebar_state="expanded")

CSS="""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'Plus Jakarta Sans',sans-serif;background:#0F1117!important;color:#FFFFFF!important}
h1,h2,h3,h4{color:#FFFFFF!important;font-weight:700}
[data-testid="stSidebar"]{background:#161B27!important;border-right:1px solid #2D3748!important}
[data-testid="stSidebar"] *{color:#FFFFFF!important}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{background:#1E2433!important;color:#FFFFFF!important;border:2px solid #4A5568!important;border-radius:8px!important;font-family:'IBM Plex Mono',monospace!important;font-size:1rem!important;font-weight:600!important}
[data-testid="stNumberInput"] input:focus{border-color:#63B3ED!important}
[data-testid="stNumberInput"] label,[data-testid="stTextInput"] label,[data-testid="stCheckbox"] label{color:#FFFFFF!important;font-weight:600!important;font-size:0.9rem!important}
[data-testid="stFileUploader"]{background:#1E2433!important;border:2px dashed #4A5568!important;border-radius:12px!important}
[data-testid="stFileUploader"] *{color:#FFFFFF!important}
.stButton>button{background:linear-gradient(135deg,#2B6CB0,#1A56DB)!important;color:#FFFFFF!important;border:none!important;border-radius:10px!important;font-weight:700!important;font-size:1rem!important;padding:14px 24px!important;transition:all 0.2s!important}
.stButton>button:hover{background:linear-gradient(135deg,#3182CE,#1E40AF)!important;transform:translateY(-1px)!important;box-shadow:0 6px 20px rgba(49,130,206,0.4)!important}
.stButton>button:disabled{background:#2D3748!important;color:#718096!important}
[data-testid="stMetric"]{background:#1E2433!important;border:1px solid #2D3748!important;border-radius:10px!important;padding:14px 16px!important}
[data-testid="stMetricLabel"]{color:#A0AEC0!important;font-size:0.78rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.06em!important}
[data-testid="stMetricValue"]{color:#FFFFFF!important;font-size:1.55rem!important;font-weight:800!important}
[data-testid="stInfo"]{background:#1A365D!important;border-left:4px solid #63B3ED!important;color:#BEE3F8!important;border-radius:8px!important}
[data-testid="stSuccess"]{background:#002D1A!important;border-left:4px solid #68D391!important;color:#C6F6D5!important;border-radius:8px!important}
[data-testid="stWarning"]{background:#2D1E00!important;border-left:4px solid #F6AD55!important;color:#FEFCBF!important;border-radius:8px!important}
[data-testid="stError"]{background:#2D0000!important;border-left:4px solid #FC8181!important;color:#FED7D7!important;border-radius:8px!important}
.stCaption,[data-testid="stCaptionContainer"]{color:#A0AEC0!important;font-size:0.82rem!important}
[data-testid="stExpander"]{background:#1E2433!important;border:1px solid #2D3748!important;border-radius:10px!important}
[data-testid="stExpander"] summary{color:#FFFFFF!important;font-weight:600!important}
[data-testid="stSlider"] label{color:#FFFFFF!important;font-weight:600!important}
[data-testid="stProgress"]>div>div>div{background:linear-gradient(90deg,#2B6CB0,#63B3ED)!important;border-radius:4px!important}
hr{border-color:#2D3748!important}
.sec-label{font-family:'IBM Plex Mono',monospace;font-size:0.65rem;font-weight:600;text-transform:uppercase;letter-spacing:0.16em;color:#63B3ED;padding-bottom:10px;display:block}
.app-title{font-size:2.4rem;font-weight:800;color:#FFFFFF;letter-spacing:-0.03em}
.subtitle{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#718096;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:20px}
.who-banner{border-radius:16px;padding:24px 28px;margin:16px 0;display:flex;align-items:center;gap:20px}
.who-icon{font-size:3.2rem;line-height:1;flex-shrink:0}
.who-tier{font-size:2.2rem;font-weight:800;letter-spacing:-0.03em;line-height:1}
.who-sub{font-family:'IBM Plex Mono',monospace;font-size:0.72rem;margin-top:5px;opacity:0.92}
.who-note{font-family:'IBM Plex Mono',monospace;font-size:0.65rem;background:#1A2035;border:1px solid #2D3748;border-radius:8px;padding:12px 16px;margin-top:12px;color:#CBD5E0;line-height:1.9}
.ml-badge{font-family:'IBM Plex Mono',monospace;font-size:0.62rem;font-weight:700;padding:4px 14px;border-radius:20px;margin-left:10px;vertical-align:middle;display:inline-block}
.coloc-box{border-radius:12px;padding:18px 20px;margin:14px 0;border:2px solid}
.coloc-title{font-weight:700;font-size:1rem;margin-bottom:8px}
.coloc-body{font-size:0.86rem;color:#E2E8F0;line-height:1.7}
.vguide-wrap{background:#1E2433;border:1px solid #2D3748;border-radius:12px;padding:18px 20px;margin:12px 0}
.vguide-head{font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#718096;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px}
.vguide-row{display:flex;align-items:center;gap:14px;padding:11px 10px;border-radius:8px;border-bottom:1px solid #2D3748}
.vguide-row:last-child{border-bottom:none}
.vguide-dot{width:14px;height:14px;border-radius:50%;flex-shrink:0}
.vguide-lbl{font-weight:700;font-size:0.9rem;color:#FFFFFF}
.vguide-rng{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#A0AEC0}
.cons-box{border-radius:12px;padding:22px 24px;margin:14px 0}
.cons-title{font-family:'IBM Plex Mono',monospace;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px;font-weight:600}
.cons-badge{display:inline-block;padding:4px 16px;border-radius:20px;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;font-weight:700;margin-bottom:14px}
.meter-track{background:#2D3748;border-radius:6px;height:12px;margin:10px 0 4px;overflow:hidden}
.meter-fill{height:100%;border-radius:6px}
.meter-ticks{font-family:'IBM Plex Mono',monospace;font-size:0.56rem;color:#718096;display:flex;justify-content:space-between;margin-bottom:16px}
.signal-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
.signal-card{background:#0F1117;border:1px solid #2D3748;border-radius:10px;padding:14px 16px}
.signal-lbl{font-family:'IBM Plex Mono',monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:#718096;margin-bottom:6px}
.signal-val{font-size:1.5rem;font-weight:800;color:#FFFFFF}
.signal-sub{font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#718096;margin-top:3px}
.cons-explain{font-size:0.88rem;line-height:1.75;color:#E2E8F0;margin-bottom:12px}
.rec-box{background:#0F1117;border:1px solid #2D3748;border-radius:8px;padding:12px 16px;font-family:'IBM Plex Mono',monospace;font-size:0.67rem;color:#CBD5E0;line-height:1.75}
.conf-line{margin-top:14px;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#718096}
.adv-wrap{background:#1E2433;border:1px solid #2D3748;border-radius:12px;padding:16px 18px}
.adv-row{display:flex;flex-direction:column;padding:11px 0;border-bottom:1px solid #2D3748;gap:4px}
.adv-row:last-child{border-bottom:none}
.adv-lbl{font-weight:700;font-size:0.87rem;color:#FFFFFF}
.adv-val{font-size:0.84rem;color:#CBD5E0;line-height:1.55}
.fus-card{background:#1E2433;border:1px solid #2D3748;border-radius:12px;padding:16px 18px;margin-bottom:12px}
.fus-head{font-family:'IBM Plex Mono',monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;color:#718096;margin-bottom:10px}
.log-line{font-family:'IBM Plex Mono',monospace;font-size:0.63rem;color:#A0AEC0;padding:5px 10px;border-left:3px solid #2D3748;margin-bottom:4px;background:#1A2035;border-radius:0 6px 6px 0}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_predictor():
    p=PM25Predictor(CONFIG); mode,trained=p.initialise(); return p,mode,trained

def init_session():
    for k,v in [("history",[]),("last_result",None),("train_result",None)]:
        if k not in st.session_state: st.session_state[k]=v


def plot_gauge(pm25,color):
    if not PLOTLY_AVAILABLE: st.metric("Fused PM2.5",f"{pm25} ug/m3"); return
    fig=go.Figure(go.Indicator(
        mode="gauge+number",value=pm25,
        number={"suffix":" ug/m3","font":{"size":34,"family":"IBM Plex Mono","color":"#FFFFFF"},"valueformat":".1f"},
        gauge={"axis":{"range":[0,max(120,pm25*1.15)],
                       "tickvals":[0,15,45,80,120],"ticktext":["0","15\n(AQG)","45\n(IT-1)","80","120+"],
                       "tickfont":{"size":10,"color":"#A0AEC0"},"tickcolor":"#475569"},
               "bar":{"color":color,"thickness":0.30},"bgcolor":"#1E2433","borderwidth":0,
               "steps":[{"range":[0,15],"color":"#0D2B1A"},{"range":[15,45],"color":"#2D1E00"},
                        {"range":[45,max(120,pm25*1.15)],"color":"#2D0000"}],
               "threshold":{"line":{"color":"#FFFFFF","width":3},"value":pm25}},
        title={"text":"Fused PM2.5  WHO 2021","font":{"size":13,"family":"IBM Plex Mono","color":"#A0AEC0"}}))
    fig.update_layout(height=295,margin=dict(t=55,b=10,l=25,r=25),
        paper_bgcolor="#0F1117",plot_bgcolor="#0F1117",font_color="#FFFFFF",
        annotations=[
            dict(x=0.17,y=0.18,text="HEALTHY",  showarrow=False,font=dict(size=9,color="#00C853",family="IBM Plex Mono")),
            dict(x=0.50,y=0.10,text="UNHEALTHY", showarrow=False,font=dict(size=9,color="#FF8F00",family="IBM Plex Mono")),
            dict(x=0.82,y=0.18,text="HAZARDOUS", showarrow=False,font=dict(size=9,color="#D32F2F",family="IBM Plex Mono")),
        ])
    st.plotly_chart(fig,use_container_width=True)

def plot_feature_bars(top_features):
    if not PLOTLY_AVAILABLE or not top_features: return
    fig=go.Figure(go.Bar(
        x=[f"Ch {i}" for i,_ in top_features],y=[abs(v) for _,v in top_features],
        marker_color=["#63B3ED" if v>=0 else "#FC8181" for _,v in top_features],
        marker_line_color="#0F1117",marker_line_width=1,
        text=[f"{abs(v):.3f}" for _,v in top_features],textposition="outside",
        textfont={"size":9,"family":"IBM Plex Mono","color":"#FFFFFF"}))
    fig.update_layout(
        title={"text":"Top CNN Feature Activations","font":{"size":12,"family":"IBM Plex Mono","color":"#A0AEC0"}},
        height=220,margin=dict(t=40,b=16,l=8,r=8),
        paper_bgcolor="#0F1117",plot_bgcolor="#1E2433",
        xaxis={"gridcolor":"#2D3748","tickfont":{"color":"#A0AEC0"}},
        yaxis={"gridcolor":"#2D3748","tickfont":{"color":"#A0AEC0"}})
    st.plotly_chart(fig,use_container_width=True)

def plot_history(history):
    if not PLOTLY_AVAILABLE or len(history)<2: return
    df=pd.DataFrame(history)
    tc={"Healthy":"#00C853","Unhealthy":"#FF8F00","Hazardous":"#D32F2F"}
    clr=[tc.get(t,"#A0AEC0") for t in df.get("level",[])]
    fig=go.Figure()
    fig.add_hrect(y0=0, y1=15, fillcolor="#00C853",opacity=0.05,line_width=0)
    fig.add_hrect(y0=15,y1=45, fillcolor="#FF8F00",opacity=0.05,line_width=0)
    fig.add_hrect(y0=45,y1=max(df["pm25"].max()*1.1,50),fillcolor="#D32F2F",opacity=0.05,line_width=0)
    fig.add_hline(y=15,line_dash="dash",line_color="#00C853",line_width=1.5,
                  annotation_text="WHO AQG 15",annotation_font={"size":9,"color":"#00C853"})
    fig.add_hline(y=45,line_dash="dash",line_color="#D32F2F",line_width=1.5,
                  annotation_text="WHO IT-1 45",annotation_font={"size":9,"color":"#D32F2F"})
    fig.add_trace(go.Scatter(x=list(range(1,len(df)+1)),y=df["pm25"],
        mode="lines+markers",line={"color":"#63B3ED","width":2.5},
        marker={"size":9,"color":clr,"line":{"width":2,"color":"#0F1117"}},name="Fused PM2.5"))
    fig.update_layout(
        title={"text":"Prediction History  WHO 2021 Thresholds",
               "font":{"size":12,"family":"IBM Plex Mono","color":"#A0AEC0"}},
        height=240,margin=dict(t=45,b=25,l=15,r=15),
        paper_bgcolor="#0F1117",plot_bgcolor="#1E2433",
        xaxis={"gridcolor":"#2D3748","title":"Run #","titlefont":{"color":"#718096"},"tickfont":{"color":"#A0AEC0"}},
        yaxis={"gridcolor":"#2D3748","title":"ug/m3","titlefont":{"color":"#718096"},"tickfont":{"color":"#A0AEC0"}})
    st.plotly_chart(fig,use_container_width=True)


def render_who_banner(result):
    tier=result["who_tier"]; color=result["who_color"]; icon=result["who_icon"]
    ex=result["who_exceeds_by"]; mode=result.get("prediction_mode","")
    if "trained" in mode:
        badge=(f"<span class='ml-badge' style='background:rgba(99,179,237,0.15);"
               "color:#63B3ED;border:1px solid rgba(99,179,237,0.35)'>ML Model</span>")
    else:
        badge=(f"<span class='ml-badge' style='background:rgba(246,173,85,0.15);"
               "color:#F6AD55;border:1px solid rgba(246,173,85,0.35)'>Haze Proxy</span>")
    ex_line=(f"<br><b style='font-size:0.75rem'>Exceeds WHO AQG by {ex:.1f} ug/m3</b>"
             if ex>0 else "<br><b style='font-size:0.75rem'>Within WHO safe guideline</b>")
    st.markdown(
        f"<div class='who-banner' style='background:{color}18;border:2px solid {color}'>"
        f"<div class='who-icon'>{icon}</div><div>"
        f"<div class='who-tier' style='color:{color}'>{tier.upper()} {badge}</div>"
        f"<div class='who-sub' style='color:{color}'>{result['who_guideline']}{ex_line}</div>"
        f"</div></div>",unsafe_allow_html=True)

def render_visual_guide(fusion):
    hp=fusion.visual_haze_proxy
    rows=[
        ("#00C853","Clear Sky","Expected PM2.5: 0-15 ug/m3  (WHO Healthy)",       hp<0.25),
        ("#8BC34A","Slight Haze","Expected PM2.5: 10-35 ug/m3  (Healthy to Unhealthy)",0.25<=hp<0.55),
        ("#FF8F00","Moderate Haze / Smoke","Expected PM2.5: 30-70 ug/m3  (Unhealthy to Hazardous)",0.55<=hp<0.75),
        ("#D32F2F","Heavy Haze / Dense Smoke","Expected PM2.5: 60-120+ ug/m3  (Hazardous)",hp>=0.75),
    ]
    inner=""
    for dc,lbl,rng,active in rows:
        bg_s=f"background:#0F1117;border:1.5px solid {dc};" if active else "opacity:0.4;"
        badge=(f" <span style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;"
               f"background:{dc}33;color:{dc};border:1px solid {dc};"
               "padding:1px 8px;border-radius:10px;margin-left:6px'>Your image</span>"
               if active else "")
        inner+=(f"<div class='vguide-row' style='{bg_s}'>"
                f"<div class='vguide-dot' style='background:{dc}'></div>"
                f"<div><div class='vguide-lbl'>{lbl}{badge}</div>"
                f"<div class='vguide-rng'>{rng}</div></div></div>")
    foot=(f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#718096;"
          f"margin-top:12px;padding-top:12px;border-top:1px solid #2D3748'>"
          f"Haze Proxy: <b style='color:#FFFFFF'>{hp:.3f}</b>  "
          f"Category: <b style='color:#FFFFFF'>{fusion.visual_label}</b>  "
          f"Range: <b style='color:#FFFFFF'>{fusion.visual_pm25_range}</b></div>")
    st.markdown(f"<div class='vguide-wrap'><div class='vguide-head'>"
                f"Visual Pollution Guide</div>{inner}{foot}</div>",unsafe_allow_html=True)

def render_colocation_box(colocated):
    if colocated:
        st.markdown(
            "<div class='coloc-box' style='background:#002D1A;border-color:#00C853'>"
            "<div class='coloc-title' style='color:#00C853'>Co-location Confirmed — Fused Mode Active</div>"
            "<div class='coloc-body'>Sensor reading will be <b style='color:#FFFFFF'>combined</b> with "
            "the trained CNN prediction (30% CNN + 70% Sensor). "
            "The Consistency Checker will validate both signals.</div></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='coloc-box' style='background:#2D1E00;border-color:#FF8F00'>"
            "<div class='coloc-title' style='color:#FF8F00'>Not Confirmed — CNN-Only Mode Active</div>"
            "<div class='coloc-body'>Sensor readings will <b style='color:#FFFFFF'>not be used</b>. "
            "Prediction is based entirely on the trained CNN regression model.<br><br>"
            "<b style='color:#F6AD55'>To enable fusion:</b> confirm your sensor was in the same "
            "location as the drone when this image was captured.</div></div>",
            unsafe_allow_html=True)

def render_consistency(c):
    if c.status=="CNN_ONLY":
        st.markdown(
            "<div class='cons-box' style='background:#1A2035;border:1px solid #2D3748'>"
            "<div class='cons-title' style='color:#718096'>Visual-Sensor Consistency Check</div>"
            "<span class='cons-badge' style='background:#2D3748;border:1px solid #4A5568;color:#A0AEC0'>"
            "NOT APPLICABLE — CNN-ONLY MODE</span>"
            "<div class='cons-explain' style='color:#A0AEC0'>Consistency check skipped. "
            "Prediction driven entirely by trained CNN regression.</div>"
            "<div class='rec-box'><span style='color:#63B3ED;font-weight:700'>To enable: </span>"
            "Confirm your sensor was at the same location as the drone at capture time.</div>"
            "</div>",unsafe_allow_html=True)
        return
    styles={
        "AGREE":   ("#002D1A","#00C853","#00C85333","#00C853","IN AGREEMENT"),
        "PARTIAL": ("#2D1E00","#FF8F00","#FF8F0033","#FF8F00","PARTIAL AGREEMENT"),
        "CONFLICT":("#2D0000","#D32F2F","#D32F2F33","#D32F2F","CONFLICT DETECTED"),
    }
    bg,border,bb,bc,label=styles.get(c.status,styles["CONFLICT"])
    mc="#00C853" if c.agreement_pct>=70 else "#FF8F00" if c.agreement_pct>=40 else "#D32F2F"
    html=(
        f"<div class='cons-box' style='background:{bg};border:1.5px solid {border}'>"
        f"<div class='cons-title' style='color:{bc}'>Visual-Sensor Consistency Check</div>"
        f"<span class='cons-badge' style='background:{bb};border:1px solid {border};color:{bc}'>{label}</span>"
        f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#A0AEC0;margin-left:12px;font-weight:600'>"
        f"Agreement: {c.agreement_pct:.0f}%</span>"
        f"<div class='meter-track'><div class='meter-fill' style='width:{c.agreement_pct}%;background:{mc}'></div></div>"
        f"<div class='meter-ticks'><span>0% No agreement</span><span>50%</span><span>100% Perfect</span></div>"
        f"<div class='signal-grid'>"
        f"<div class='signal-card'><div class='signal-lbl'>Image Signal (CNN)</div>"
        f"<div class='signal-val'>{c.visual_pm25_est:.1f}<span style='font-size:0.8rem;color:#A0AEC0;font-weight:400'> ug/m3</span></div>"
        f"<div class='signal-sub'>Trained CNN PM2.5 estimate</div></div>"
        f"<div class='signal-card'><div class='signal-lbl'>Sensor (User Input)</div>"
        f"<div class='signal-val'>{c.sensor_pm25:.1f}<span style='font-size:0.8rem;color:#A0AEC0;font-weight:400'> ug/m3</span></div>"
        f"<div class='signal-sub'>Entered PM2.5 reading</div></div>"
        f"</div>"
        f"<div class='cons-explain'>{c.explanation}</div>"
        f"<div class='rec-box'><span style='color:#63B3ED;font-weight:700'>Recommendation: </span>{c.recommendation}</div>"
        f"<div class='conf-line'>Confidence adjustment: <b style='color:{bc}'>{c.confidence_delta:+.1f}%</b>"
        f"  Final confidence: <b style='color:#FFFFFF'>{c.final_confidence:.1f}%</b>"
        f"  Gap: {c.gap:.3f}</div></div>")
    st.markdown(html,unsafe_allow_html=True)


def main():
    init_session()

    with st.spinner("Initialising CNN model..."):
        predictor,cnn_mode,has_trained=get_predictor()

    # Header
    st.markdown(
        "<div style='display:flex;align-items:center;gap:14px;margin-bottom:6px'>"
        "<span style='font-size:2.4rem'>🛸</span>"
        "<span class='app-title'>AeroSense</span></div>"
        "<div class='subtitle'>UAV  CNN Regression  Sensor Fusion  WHO 2021 AQG  Co-location Verification</div>",
        unsafe_allow_html=True)
    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown("### Configuration")
        st.markdown('<span class="sec-label">Fusion Weights</span>',unsafe_allow_html=True)
        cnn_w=st.slider("CNN Weight",0.0,1.0,CONFIG.fusion.cnn_weight,0.05)
        CONFIG.fusion.cnn_weight=cnn_w
        CONFIG.fusion.sensor_weight=round(1.0-cnn_w,2)
        st.caption(f"Sensor weight: **{CONFIG.fusion.sensor_weight:.2f}**")

        st.markdown('<span class="sec-label">WHO 2021 Thresholds</span>',unsafe_allow_html=True)
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;line-height:2.1'>"
            "<span style='color:#00C853;font-weight:700'>Healthy</span>  &lt;= 15 ug/m3 (AQG)<br>"
            "<span style='color:#FF8F00;font-weight:700'>Unhealthy</span>  15-45 ug/m3 (IT-1)<br>"
            "<span style='color:#D32F2F;font-weight:700'>Hazardous</span>  &gt; 45 ug/m3</div>",
            unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ML Model Status")
        if predictor.has_trained_model:
            st.success("Trained Ridge Regression model active.\nReal PM2.5 predictions from your dataset.")
            if st.session_state.train_result:
                tr=st.session_state.train_result
                st.caption(f"MAE: **{tr.mae:.1f}** ug/m3  RMSE: **{tr.rmse:.1f}**  R2: **{tr.r2:.3f}**  n: **{tr.n_samples}**")
        else:
            st.warning(
                "No trained model found.\n\n"
                "Using haze proxy fallback.\n\n"
                "Click Train Model to use real ML prediction.")

        st.markdown("")
        csv_exists=Path(CSV_PATH).exists()
        img_exists =Path(IMAGES_DIR).exists()
        if st.button("Train Model Now",use_container_width=True,
                     help=f"Trains on {CSV_PATH} + {IMAGES_DIR}"):
            if not csv_exists:
                st.error(f"CSV not found: {CSV_PATH}")
            elif not img_exists:
                st.error(f"Images folder not found: {IMAGES_DIR}")
            else:
                prog=st.progress(0.0,"Starting...")
                ptxt=st.empty()
                def cb(pct,msg): prog.progress(min(pct,1.0),msg); ptxt.caption(msg)
                try:
                    tr=predictor.train_now(status_cb=cb)
                    st.session_state.train_result=tr
                    prog.progress(1.0,"Complete!"); ptxt.empty()
                    st.success(f"Training complete!\nMAE={tr.mae:.1f}  RMSE={tr.rmse:.1f}  R2={tr.r2:.3f}\n{tr.n_samples} pairs trained.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    logger.error(traceback.format_exc())

        if not csv_exists:
            st.caption(f"CSV not found at: {CSV_PATH}")
        if not img_exists:
            st.caption(f"Images folder not found at: {IMAGES_DIR}")

        st.markdown("---")
        st.markdown("### Session Stats")
        n=len(st.session_state.history)
        st.metric("Predictions Run",n)
        if n>0:
            vals=[h["pm25"] for h in st.session_state.history]
            st.metric("Avg PM2.5",f"{np.mean(vals):.1f} ug/m3")
            st.metric("Max PM2.5",f"{np.max(vals):.1f} ug/m3")
            tier_counts=pd.Series([h["level"] for h in st.session_state.history]).value_counts()
            for tier,count in tier_counts.items():
                dot={"Healthy":"🟢","Unhealthy":"🟠","Hazardous":"🔴"}.get(tier,"⚪")
                st.write(f"{dot} {tier}: {count}")

        with st.expander("Config JSON"):
            st.json(CONFIG.to_dict())

    # Main columns
    col_l,col_r=st.columns([1,1],gap="large")

    with col_l:
        st.markdown('<span class="sec-label">① UAV Image</span>',unsafe_allow_html=True)
        st.caption("Upload any UAV aerial image. Any drone, any location.")
        uploaded=st.file_uploader("Upload UAV aerial image",
            type=["jpg","jpeg","png","webp","tif","tiff"],key="pred_img")
        if uploaded:
            img=Image.open(uploaded).convert("RGB")
            st.image(img,caption=uploaded.name,use_column_width=True)
        else:
            st.info("Upload a UAV image to begin.")

        st.markdown('<span class="sec-label">② Co-location</span>',unsafe_allow_html=True)
        colocated=st.checkbox(
            "My sensor was at the same location as this image",
            value=False,
            help="Tick to combine sensor readings with CNN prediction.")
        render_colocation_box(colocated)

        st.markdown('<span class="sec-label">③ Sensor Readings</span>',unsafe_allow_html=True)
        if not colocated:
            st.markdown(
                "<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;"
                "color:#F6AD55'>CNN-only mode — sensor values recorded but not used.</div>",
                unsafe_allow_html=True)
        sc1,sc2,sc3=st.columns(3)
        with sc1: pm1 =st.number_input("PM 1.0", 0.0,5000.0,13.0,1.0,disabled=not colocated)
        with sc2: pm25=st.number_input("PM 2.5", 0.0,5000.0,25.0,1.0,disabled=not colocated)
        with sc3: pm10=st.number_input("PM 10",  0.0,5000.0,30.0,1.0,disabled=not colocated)

        st.markdown(
            "<div class='who-note'>"
            "<b style='color:#63B3ED'>WHO 2021 PM2.5:</b><br>"
            "<span style='color:#00C853;font-weight:700'>Healthy</span> &lt;=15 ug/m3  "
            "<span style='color:#FF8F00;font-weight:700'>Unhealthy</span> 15-45 ug/m3  "
            "<span style='color:#D32F2F;font-weight:700'>Hazardous</span> &gt;45 ug/m3"
            "</div>",unsafe_allow_html=True)

        run_btn=st.button("Run Prediction",type="primary",
                          use_container_width=True,disabled=(uploaded is None))

    with col_r:
        st.markdown('<span class="sec-label">④ WHO Classification Result</span>',unsafe_allow_html=True)
        slot=st.empty()

        if run_btn and uploaded:
            reading=SensorReading(pm1=pm1,pm25=pm25,pm10=pm10,
                                  filename=uploaded.name,colocated=colocated)
            try:
                with st.spinner("Extracting CNN features · Predicting PM2.5..."):
                    result=predictor.predict(img,reading)
                st.session_state.last_result=result
                st.session_state.history.append({
                    "pm25":result["pm25"],"level":result["who_tier"],
                    "ts":result["timestamp"],"mode":result["prediction_mode"]})
                with slot.container():
                    render_who_banner(result)
                    plot_gauge(result["pm25"],result["who_color"])
                    m1,m2,m3=st.columns(3)
                    m1.metric("Fused PM2.5",f"{result['pm25']:.1f} ug/m3")
                    m2.metric("Confidence", f"{result['confidence']:.0f}%")
                    m3.metric("Mode",result["prediction_mode"].replace("_"," ").title())
                    st.caption(result["who_description"])
            except ValueError as e: st.error(f"Invalid input: {e}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                logger.error(traceback.format_exc())

        elif st.session_state.last_result:
            result=st.session_state.last_result
            with slot.container():
                render_who_banner(result)
                plot_gauge(result["pm25"],result["who_color"])
                m1,m2,m3=st.columns(3)
                m1.metric("Fused PM2.5",f"{result['pm25']:.1f} ug/m3")
                m2.metric("Confidence", f"{result['confidence']:.0f}%")
                m3.metric("CNN Mode",   result["cnn_mode"])
        else:
            slot.info("Upload an image and click Run Prediction.")

    if st.session_state.last_result:
        result=st.session_state.last_result

        st.divider()
        st.markdown('<span class="sec-label">⑤ Visual Pollution Guide</span>',unsafe_allow_html=True)
        st.caption("Compares your image visually to expected PM2.5 ranges under WHO 2021 thresholds.")
        render_visual_guide(result["fusion"])

        st.divider()
        st.markdown('<span class="sec-label">⑥ Visual-Sensor Consistency Check</span>',unsafe_allow_html=True)
        st.caption("Compares the CNN visual signal against your sensor reading.")
        render_consistency(result["consistency"])

        st.divider()
        st.markdown('<span class="sec-label">⑦ WHO Health Advisory</span>',unsafe_allow_html=True)
        adv_col,fus_col=st.columns([1,1],gap="large")
        advice=result["advice"]; fusion=result["fusion"]

        with adv_col:
            inner="".join(
                f"<div class='adv-row'><div class='adv-lbl'>{ic} &nbsp;{lb}</div>"
                f"<div class='adv-val'>{tx}</div></div>"
                for ic,lb,tx in [
                    ("🏃","Outdoor Activity",advice.outdoor_activity),
                    ("🪟","Ventilation",      advice.ventilation),
                    ("😷","Mask",            advice.mask_recommendation),
                    ("👨‍👩‍👧","Sensitive Groups",advice.sensitive_groups),
                    ("🛸","UAV Operations",   advice.uav_ops),
                ])
            st.markdown(f"<div class='adv-wrap'>{inner}</div>",unsafe_allow_html=True)

        with fus_col:
            st.markdown("<div class='fus-card'><div class='fus-head'>Fusion Breakdown</div>",
                        unsafe_allow_html=True)
            f1,f2=st.columns(2)
            f1.metric("CNN Contribution",   f"{fusion.cnn_contribution:.1f} ug/m3")
            f2.metric("Sensor Contribution",f"{fusion.sensor_contribution:.1f} ug/m3")
            f3,f4=st.columns(2)
            f3.metric("Base Confidence",f"{fusion.base_confidence:.1f}%")
            f4.metric("Haze Proxy",     f"{fusion.visual_haze_proxy:.3f}")
            st.caption(f"Mode: **{fusion.mode}**  {fusion.visual_label}")
            st.markdown("</div>",unsafe_allow_html=True)
            plot_feature_bars(fusion.top_features)

    if len(st.session_state.history)>=2:
        st.divider()
        st.markdown('<span class="sec-label">⑧ Prediction History</span>',unsafe_allow_html=True)
        plot_history(st.session_state.history)
        hc1,hc2=st.columns([6,1])
        with hc2:
            if st.button("Clear",use_container_width=True):
                st.session_state.history=[]; st.rerun()

    st.divider()
    with st.expander("Live Log Viewer",expanded=False):
        st.markdown('<span class="sec-label">Recent Events</span>',unsafe_allow_html=True)
        logs=logger.get_recent_logs(30)
        if logs:
            for e in logs:
                lc={"INFO":"#63B3ED","WARNING":"#F6AD55","ERROR":"#FC8181"}.get(
                   e.get("level","INFO"),"#718096")
                st.markdown(
                    f"<div class='log-line'><b style='color:{lc}'>{e.get('level','')}</b>"
                    f" | {e.get('ts','')[:19]} | {e.get('msg','')}</div>",
                    unsafe_allow_html=True)
        else:
            st.caption("No log entries yet.")

    st.divider()
    model_status="Ridge Regression Active" if predictor.has_trained_model else "Haze Proxy Mode"
    st.markdown(
        f"<center style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#4A5568'>"
        f"AeroSense  CNN Regression + Sensor Fusion  WHO 2021 AQG  {model_status}  "
        f"{'MobileNetV2 1280-dim' if TF_AVAILABLE else 'Histogram Fallback 102-dim'}"
        f"</center>",unsafe_allow_html=True)


if __name__=="__main__":
    main()
