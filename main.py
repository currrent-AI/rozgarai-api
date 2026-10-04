import os
import json
import psycopg
import edge_tts
from psycopg.rows import dict_row
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Awaaz API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Location na mile to Gulberg, Lahore (demo default)
DEFAULT_LAT, DEFAULT_LNG = 31.5120, 74.3460

MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"]

STT_HINT = (
    "Mera naam Aslam hai, main Gulberg Lahore mein bijli ka kaam karta hoon. "
    "electrician, plumber, painter, mistri, carpenter, bijli wala, nal ka kaam."
)

SKILLS = [
    "electrician", "plumber", "painter", "carpenter",
    "mason", "ac_technician", "driver", "cleaner", "other",
]

SKILL_DICT = """
electrician: bijli wala, bijli ka kaam, wiring, light ka kaam, electrician, بجلی, وائرنگ, الیکٹریشن
plumber: nal ka kaam, plumber, pipe, nalka, پلمبر, نل, پائپ
painter: rang, rangsaz, paint, painter, رنگ, پینٹر, پینٹ
carpenter: lakri ka kaam, carpenter, tarkhan, بڑھئی, ترکھان, لکڑی
mason: mistri, raj, rajmistri, imarat, مستری, راج, چنائی
ac_technician: ac, ac ka kaam, cooling, اے سی
driver: driver, gari chalana, ڈرائیور, گاڑی
cleaner: safai, cleaner, jharu, صفائی
AREAS = "Gulberg, Model Town, Johar Town, Garden Town, DHA, Iqbal Town, Samanabad, Township, Wapda Town, Bahria Town, Shadman, Faisal Town, Cantt, Ravi Road, Data Darbar"
"""

WORKER_SYSTEM = f"""You extract a worker profile from a spoken self-introduction.
The transcript may be Urdu script, Roman Urdu, or Punjabi, with speech-recognition mistakes.
Return ONLY a JSON object with these keys:
- "name": the person's name as spoken (string or null)
- "skill": exactly one of {SKILLS}. Use "other" if no match.
- "area": neighbourhood or city mentioned (string or null)
- "experience_years": integer or null

Skill dictionary (local words -> skill):
{SKILL_DICT}
"""

JOB_SYSTEM = f"""A customer describes a job they need done, spoken in Urdu, Roman Urdu or Punjabi.
Return ONLY a JSON object with these keys:
- "skill": exactly one of {SKILLS}. Use "other" if no match.
- "area": neighbourhood or city where the work is (string or null)

Skill dictionary (local words -> skill):
{SKILL_DICT}
"""


def get_db():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME", "postgres"),
        sslmode="require",
        row_factory=dict_row,
        autocommit=True,
        prepare_threshold=None,
    )


def stt(data: bytes, lang: str) -> str:
    if len(data) < 1000:
        raise HTTPException(400, "Audio bohot chhoti hai")
    try:
        res = client.audio.transcriptions.create(
            file=("voice.webm", data),
            model="whisper-large-v3",
            language=lang,
            prompt=STT_HINT,
            response_format="json",
            temperature=0.0,
        )
        return res.text.strip()
    except Exception as e:
        raise HTTPException(502, f"STT fail: {e}")


def llm_json(system: str, text: str) -> dict:
    last_err = None
    for model in MODELS:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            p = json.loads(r.choices[0].message.content)
            print(f"[LLM ok] {model}: {p}")
            return p
        except Exception as e:
            print(f"[LLM fail] {model}: {e}")
            last_err = e
    raise HTTPException(502, f"LLM fail: {last_err}")


def clean_skill(skill) -> str:
    return skill if skill in SKILLS else "other"


def insert_worker(profile: dict, text: str, lang: str, lat, lng) -> str:
    qlat = lat if lat is not None else DEFAULT_LAT
    qlng = lng if lng is not None else DEFAULT_LNG
    with get_db() as conn:
        row = conn.execute(
            """insert into workers (name, skill, area, experience_years, transcript, lang, geo)
               values (%s, %s, %s, %s, %s, %s, st_setsrid(st_makepoint(%s, %s), 4326)::geography)
               returning id::text as id""",
            (profile.get("name"), profile["skill"], profile.get("area"),
             profile.get("experience_years"), text, lang, qlng, qlat),
        ).fetchone()
    return row["id"]


def nearest_workers(skill: str, lat: float, lng: float, limit: int = 3):
    sql = """
      select id::text as id, name, skill, area, experience_years,
             st_y(geo::geometry) as lat, st_x(geo::geometry) as lng,
             round((st_distance(geo, st_setsrid(st_makepoint(%(lng)s, %(lat)s), 4326)::geography) / 1000)::numeric, 2)::float8 as distance_km
      from workers
      where skill = %(skill)s and geo is not null
      order by geo <-> st_setsrid(st_makepoint(%(lng)s, %(lat)s), 4326)::geography
      limit %(limit)s
    """
    with get_db() as conn:
        return conn.execute(sql, {"lat": lat, "lng": lng, "skill": skill, "limit": limit}).fetchall()


@app.get("/health")
def health():
    try:
        with get_db() as conn:
            n = conn.execute("select count(*) as n from workers").fetchone()["n"]
        return {"ok": True, "db": True, "workers": n}
    except Exception as e:
        return {"ok": True, "db": False, "error": str(e)}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), lang: str = Form("ur")):
    return {"text": stt(await audio.read(), lang), "lang": lang}


@app.post("/register-worker")
async def register_worker(
    audio: UploadFile = File(...),
    lang: str = Form("ur"),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
    save: str = Form("1"),  # "0" = sirf profile banao, confirm ke baad save hoga
):
    text = stt(await audio.read(), lang)
    p = llm_json(WORKER_SYSTEM, text)
    profile = {
        "name": p.get("name"),
        "skill": clean_skill(p.get("skill")),
        "area": p.get("area"),
        "experience_years": p.get("experience_years"),
    }
    if save == "1":
        profile["id"] = insert_worker(profile, text, lang, lat, lng)
    return {"text": text, "profile": profile}


class SaveIn(BaseModel):
    name: str | None = None
    skill: str = "other"
    area: str | None = None
    experience_years: int | None = None
    transcript: str | None = None
    lang: str = "ur"
    lat: float | None = None
    lng: float | None = None


@app.post("/save-worker")
def save_worker(b: SaveIn):
    profile = {
        "name": b.name,
        "skill": clean_skill(b.skill),
        "area": b.area,
        "experience_years": b.experience_years,
    }
    profile["id"] = insert_worker(profile, b.transcript or "", b.lang, b.lat, b.lng)
    return {"profile": profile}


@app.get("/workers")
def list_workers():
    with get_db() as conn:
        rows = conn.execute(
            """select id::text as id, name, skill, area, experience_years,
                      st_y(geo::geometry) as lat, st_x(geo::geometry) as lng
               from workers where geo is not null
               order by created_at desc limit 500"""
        ).fetchall()
    return {"workers": rows}


@app.post("/post-job")
async def post_job(
    audio: UploadFile = File(...),
    lang: str = Form("ur"),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
):
    text = stt(await audio.read(), lang)
    p = llm_json(JOB_SYSTEM, text)
    job = {"skill": clean_skill(p.get("skill")), "area": p.get("area")}
    qlat = lat if lat is not None else DEFAULT_LAT
    qlng = lng if lng is not None else DEFAULT_LNG
    with get_db() as conn:
        row = conn.execute(
            """insert into jobs (skill, area, transcript, geo)
               values (%s, %s, %s, st_setsrid(st_makepoint(%s, %s), 4326)::geography)
               returning id::text as id""",
            (job["skill"], job["area"], text, qlng, qlat),
        ).fetchone()
    job["id"] = row["id"]
    return {"text": text, "job": job, "matches": nearest_workers(job["skill"], qlat, qlng)}


@app.get("/match")
def match(skill: str, lat: float = DEFAULT_LAT, lng: float = DEFAULT_LNG, limit: int = 3):
    return {"matches": nearest_workers(skill, lat, lng, limit)}


class SpeakIn(BaseModel):
    text: str
    voice: str = "ur-PK-UzmaNeural"


@app.post("/speak")
async def speak(body: SpeakIn):
    try:
        buf = bytearray()
        async for chunk in edge_tts.Communicate(body.text, body.voice).stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        if not buf:
            raise ValueError("khali audio")
        return Response(bytes(buf), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(502, f"TTS fail: {e}")