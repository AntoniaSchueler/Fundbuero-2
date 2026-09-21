import io
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(
    page_title="Fundbüro am Katharineum neu",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CATEGORIES = ["Trinkflaschen", "Schuhe", "T-Shirts", "Federtaschen", "Sonstiges"]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "fundbuero.db"
MAX_UPLOAD_MB = 8
MAX_IMAGE_SIZE = 1600

# Vortrainiertes Ultralytics-Klassifizierungsmodell.
# Die Gewichte werden beim ersten Start automatisch geladen.
MODEL_NAME = "yolo26n-cls.pt"

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');
    :root { --green:#145c2a; --green-light:#eaf4ec; --green-mid:#2e7d46; --cream:#fbfaf6; --ink:#18321f; }
    .stApp { background:var(--cream); color:var(--ink); }
    .block-container { max-width:1180px; padding-top:2rem; padding-bottom:4rem; }
    h1,h2,h3 { font-family:'Libre Baskerville',Georgia,serif !important; color:var(--green) !important; }
    .brand-title { font-family:'Libre Baskerville',Georgia,serif; font-style:italic; font-size:clamp(2.2rem,5vw,4.2rem); line-height:1.1; color:var(--green); margin:1rem 0 2.2rem; }
    .top-rule { height:3px; background:var(--green); margin:.5rem 0 1.5rem; }
    .hero-card { border:2px solid var(--green); padding:2rem; background:rgba(255,255,255,.45); min-height:250px; display:flex; flex-direction:column; justify-content:center; }
    .hero-card h2 { margin-top:0; font-size:2rem; }
    .fund-category { color:var(--green); font-family:'Libre Baskerville',Georgia,serif; font-weight:700; margin-top:.7rem; }
    .fund-location { color:#526057; font-size:.9rem; margin-top:.2rem; }
    div.stButton > button { border:2px solid var(--green); color:var(--green); background:transparent; border-radius:0; min-height:3.2rem; font-family:'Libre Baskerville',Georgia,serif; font-size:1.05rem; }
    div.stButton > button:hover { color:white; background:var(--green); border-color:var(--green); }
    .primary-button div.stButton > button { background:var(--green); color:white; }
    [data-testid="stFileUploader"] { border:1.5px dashed var(--green); padding:.5rem; background:white; }

    /* Eingabefelder sollen nach dem Ausfüllen nicht rot markiert werden. */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        border-color:#d0d7d2 !important;
        box-shadow:none !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color:var(--green) !important;
        box-shadow:0 0 0 1px var(--green) !important;
    }
    [data-testid="stTextInput"] [aria-invalid="true"] {
        border-color:#d0d7d2 !important;
        box-shadow:none !important;
    }

    .footer { text-align:center; color:var(--green-mid); margin-top:4rem; font-family:'Libre Baskerville',Georgia,serif; font-style:italic; }
    </style>
    """,
    unsafe_allow_html=True,
)


def utc_now():
    return datetime.now(timezone.utc)


def connect_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fundstuecke (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            location TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'verfuegbar',
            collected_at TEXT
        )
        """
    )
    conn.commit()
    return conn


@st.cache_resource
def get_db():
    return connect_db()


def cleanup_old_items(conn):
    now = utc_now()
    week_ago = now - timedelta(days=7)
    year_ago = now - timedelta(days=365)
    rows = conn.execute(
        "SELECT id, image_filename, status, created_at, collected_at FROM fundstuecke"
    ).fetchall()

    for row in rows:
        remove = False
        if row["status"] == "abgeholt" and row["collected_at"]:
            try:
                remove = datetime.fromisoformat(row["collected_at"]) < week_ago
            except ValueError:
                pass
        elif row["status"] != "abgeholt":
            try:
                remove = datetime.fromisoformat(row["created_at"]) < year_ago
            except ValueError:
                pass

        if remove:
            image_path = IMAGE_DIR / row["image_filename"]
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                pass
            conn.execute("DELETE FROM fundstuecke WHERE id = ?", (row["id"],))

    conn.commit()


@st.cache_resource
def get_model():
    return YOLO(MODEL_NAME)


# Übersetzung einiger allgemeiner ImageNet-Bezeichnungen in die
# Fundbüro-Kategorien. Nicht passende Klassen werden zu "Sonstiges".
KEYWORDS = {
    "Trinkflaschen": (
        "water bottle", "bottle", "pop bottle", "beer bottle",
        "wine bottle", "pill bottle", "milk can"
    ),
    "Schuhe": (
        "running shoe", "shoe", "sneaker", "clog", "sandal",
        "cowboy boot", "boot", "loafer", "slipper", "footwear"
    ),
    "T-Shirts": (
        "jersey", "shirt", "polo shirt", "sweatshirt", "maillot"
    ),
    "Federtaschen": (
        "pencil box", "pencil case"
    ),
}


def map_to_category(model_label):
    label = str(model_label).lower().strip()
    for category, keywords in KEYWORDS.items():
        if any(keyword in label for keyword in keywords):
            return category
    return "Sonstiges"


def predict_category(image):
    model = get_model()
    results = model.predict(source=image, verbose=False)
    result = results[0]

    if result.probs is None:
        return "Sonstiges", 0.0, "Keine Klassifikation verfügbar"

    top1 = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    model_label = result.names[top1]
    category = map_to_category(model_label)

    return category, confidence, model_label


def prepare_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIZE:
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=True)
    return image, output.getvalue()


def insert_item(conn, category, location, image_bytes):
    item_id = uuid.uuid4().hex
    filename = f"{item_id}.jpg"
    image_path = IMAGE_DIR / filename
    image_path.write_bytes(image_bytes)

    try:
        conn.execute(
            "INSERT INTO fundstuecke "
            "(id, category, location, image_filename, created_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (item_id, category, location.strip(), filename,
             utc_now().isoformat(), "verfuegbar"),
        )
        conn.commit()
    except Exception:
        image_path.unlink(missing_ok=True)
        raise


def get_available_items(conn, category=None):
    if category and category != "Alle":
        return conn.execute(
            "SELECT * FROM fundstuecke WHERE status='verfuegbar' "
            "AND category=? ORDER BY created_at DESC",
            (category,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM fundstuecke WHERE status='verfuegbar' "
        "ORDER BY created_at DESC"
    ).fetchall()


def mark_collected(conn, item_id):
    conn.execute(
        "UPDATE fundstuecke SET status='abgeholt', collected_at=? "
        "WHERE id=? AND status='verfuegbar'",
        (utc_now().isoformat(), item_id),
    )
    conn.commit()


def admin_code():
    try:
        return str(st.secrets["ADMIN_CODE"])
    except Exception:
        return "sekretariat123"


conn = get_db()
cleanup_old_items(conn)

if "page" not in st.session_state:
    st.session_state.page = "start"

st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-title">Fundbüro am<br> Katharineum</div>',
    unsafe_allow_html=True,
)

nav1, nav2, nav3 = st.columns(3)
with nav1:
    if st.button("Startseite", use_container_width=True):
        st.session_state.page = "start"
        st.rerun()
with nav2:
    if st.button("Gefunden", use_container_width=True):
        st.session_state.page = "gefunden"
        st.rerun()
with nav3:
    if st.button("Verwaltung", use_container_width=True):
        st.session_state.page = "verwaltung"
        st.rerun()

st.markdown("---")


def render_start():
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="hero-card"><h2>Verloren</h2>'
            '<p>Du suchst einen Gegenstand? Durchsuche die aktuell '
            'gefundenen Sachen nach Kategorie.</p></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Gefundene Sachen ansehen", use_container_width=True):
            st.session_state.page = "gefunden"
            st.rerun()

    with right:
        st.markdown(
            '<div class="hero-card"><h2>Gefunden</h2>'
            '<p>Du hast etwas gefunden? Lade ein Foto hoch. Die KI hilft '
            'bei der Einordnung.</p></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Fundstück melden", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()

    st.markdown(
        '<div class="footer">Ein verlorener Test Gegenstand findet vielleicht '
        'seinen Weg zurück.</div>',
        unsafe_allow_html=True,
    )


def render_upload():
    st.header("Fundstück melden")
    st.write(
        "Lade ein Foto hoch. Das vortrainierte KI-Modell schlägt eine "
        "Kategorie vor. Du kannst sie vor dem Speichern korrigieren."
    )
    uploaded = st.file_uploader(
        "Foto auswählen",
        type=["jpg", "jpeg", "png"],
        help=f"Maximal {MAX_UPLOAD_MB} MB.",
    )
    if uploaded is None:
        st.info("Bitte zuerst ein Foto auswählen.")
        return
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Das Bild darf höchstens {MAX_UPLOAD_MB} MB groß sein.")
        return

    try:
        image, image_bytes = prepare_image(uploaded)
        predicted, confidence, model_label = predict_category(image)
    except Exception as exc:
        st.error("Das Bild oder das KI-Modell konnte nicht verarbeitet werden.")
        st.exception(exc)
        return

    left, right = st.columns(2, gap="large")
    with left:
        st.image(image, caption="Vorschau", use_container_width=True)

    with right:
        st.subheader("KI-Ergebnis")
        st.write(f"**Erkannte Kategorie:** {predicted}")
        st.caption(f"Vortrainiertes Modell erkannte: {model_label}")
        st.progress(min(max(confidence, 0.0), 1.0))

        category = st.selectbox(
            "Kategorie bestätigen oder ändern",
            CATEGORIES,
            index=CATEGORIES.index(predicted),
        )
        location = st.text_input(
            "Fundort",
            placeholder="z. B. Sporthalle, Pausenhof, Raum 204 …",
            max_chars=200,
        )

        st.markdown('<div class="primary-button">', unsafe_allow_html=True)
        save = st.button("Fundstück speichern", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if save:
            if not location.strip():
                st.warning("Bitte gib einen kurzen Fundort an.")
            else:
                try:
                    insert_item(conn, category, location, image_bytes)
                    st.success("Das Fundstück wurde gespeichert.")
                    st.session_state.page = "gefunden"
                    st.rerun()
                except Exception as exc:
                    st.error("Das Fundstück konnte nicht gespeichert werden.")
                    st.exception(exc)


def render_found():
    st.header("Gefundene Gegenstände")
    selected = st.selectbox("Kategorie", ["Alle"] + CATEGORIES)
    items = get_available_items(conn, selected)
    if not items:
        st.info("Aktuell wurden keine passenden Fundstücke gefunden.")
        return

    cols = st.columns(3, gap="large")
    for i, item in enumerate(items):
        with cols[i % 3]:
            image_path = IMAGE_DIR / item["image_filename"]
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            st.markdown(
                f'<div class="fund-category">{item["category"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="fund-location">Fundort: {item["location"]}</div>',
                unsafe_allow_html=True,
            )
            try:
                dt = datetime.fromisoformat(item["created_at"])
                st.caption(f"Gefunden am {dt.astimezone().strftime('%d.%m.%Y')}")
            except Exception:
                pass


def render_admin():
    st.header("Verwaltung")
    st.write(
        "Dieser Bereich ist für das Sekretariat. Hier können Fundstücke "
        "als abgeholt markiert werden."
    )
    code = st.text_input("Verwaltungscode", type="password")
    if not code:
        st.info("Bitte Verwaltungscode eingeben.")
        return
    if code != admin_code():
        st.error("Der Verwaltungscode ist nicht korrekt.")
        return

    st.success("Verwaltungsbereich geöffnet.")
    items = get_available_items(conn)
    if not items:
        st.info("Es sind aktuell keine offenen Fundstücke vorhanden.")
        return

    for item in items:
        with st.container(border=True):
            left, middle, right = st.columns([1, 2, 1])
            with left:
                image_path = IMAGE_DIR / item["image_filename"]
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
            with middle:
                st.subheader(item["category"])
                st.write(f"**Fundort:** {item['location']}")
            with right:
                confirm = st.checkbox(
                    "Als abgeholt markieren", key=f"confirm_{item['id']}"
                )
                if confirm and st.button(
                    "Abholung speichern",
                    key=f"collect_{item['id']}",
                    use_container_width=True,
                ):
                    mark_collected(conn, item["id"])
                    st.success("Erledigt.")
                    st.rerun()


if st.session_state.page == "start":
    render_start()
elif st.session_state.page == "gefunden":
    render_found()
elif st.session_state.page == "upload":
    render_upload()
elif st.session_state.page == "verwaltung":
    render_admin()
else:
    st.session_state.page = "start"
    render_start()
