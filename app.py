"""
MTG Color Classifier: upload a picture, get the five Magic colors the model sees in it.

The model lives on Hugging Face (paularoni/mtg-color-classifier) as one bundle:
weights + backbone name + per-color thresholds + preprocessing settings. A new
model is swapped in there (promote_model.py in the research repo); this app picks
it up on its next restart, with no code change.

Run locally:
    streamlit run app.py
"""

from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from torchvision import transforms as T

HF_REPO = "paularoni/mtg-color-classifier"
BUNDLE_NAME = "model_bundle.pt"
MAX_UPLOAD_MB = 10
NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}

# Mana colors as chart colors, checked for color-blind separation and contrast per theme.
# White is drawn gold (a white bar vanishes on a white page), black as a dark plum.
BAR_COLORS = {
    "light": {"W": "#B8860B", "U": "#3B95DB", "B": "#6A3F8E", "R": "#A8281F", "G": "#2E9A5E"},
    "dark": {"W": "#B08A1E", "U": "#3F94D6", "B": "#7E52A8", "R": "#B83A33", "G": "#43A86A"},
}
INK = {"light": "#31333F", "dark": "#FAFAFA"}  # Streamlit's text colors

SRGB = ImageCms.createProfile("sRGB")
torch.set_num_threads(2)  # small shared machine on Streamlit Cloud


# ================================================================ model
@st.cache_resource(show_spinner="Loading the model (after a quiet spell this takes a minute) ...")
def load_model() -> tuple[torch.nn.Module, dict]:
    """Download the live bundle once per app start, not per visitor."""
    path = hf_hub_download(HF_REPO, BUNDLE_NAME)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    net = timm.create_model(bundle["backbone"], pretrained=False, num_classes=len(bundle["classes"]))
    net.load_state_dict(bundle["state_dict"])
    info = {k: v for k, v in bundle.items() if k != "state_dict"}
    return net.eval(), info


# ---- copied from mtg-color scripts/promote_model.py: keep identical.
# It reproduces how the training images were prepared; the promote script's
# parity check proves this chain matches the model's test predictions.
def to_srgb(im: Image.Image) -> Image.Image:
    """Same as scripts/resize_images.py: color-manage ICC-tagged images into sRGB."""
    icc = im.info.get("icc_profile")
    if icc:
        if im.mode not in ("RGB", "CMYK", "L"):
            im = im.convert("RGB")
        src_profile = ImageCms.ImageCmsProfile(BytesIO(icc))
        return ImageCms.profileToProfile(im, src_profile, SRGB, outputMode="RGB",
                                         renderingIntent=ImageCms.Intent.PERCEPTUAL)
    return im.convert("RGB")


def eval_transform(pp: dict):
    """The notebook's eval_transform: torchvision Resize on a PIL image (PIL bilinear)."""
    return T.Compose([T.Resize((pp["img_size"], pp["img_size"])), T.ToTensor(),
                      T.Normalize(pp["mean"], pp["std"])])


def preprocess(im: Image.Image, pp: dict) -> torch.Tensor:
    """Any uploaded image -> model input, the way the app will do it."""
    im = to_srgb(im).resize((pp["cache_size"], pp["cache_size"]), Image.Resampling.LANCZOS)
    buf = BytesIO()  # the cached training images were JPEGs: same artifacts, same predictions
    im.save(buf, "JPEG", quality=pp["cache_jpeg_quality"], subsampling=pp["cache_jpeg_subsampling"])
    buf.seek(0)
    return eval_transform(pp)(Image.open(buf).convert("RGB"))
# ---- end of copy


def open_upload(data: bytes) -> Image.Image:
    """Uploaded bytes -> PIL image, upright, with any transparency flattened onto white."""
    im = Image.open(BytesIO(data))
    im = ImageOps.exif_transpose(im)  # phone photos store rotation in EXIF
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        icc = im.info.get("icc_profile")
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.getchannel("A"))
        if icc:
            flat.info["icc_profile"] = icc
        im = flat
    return im


@torch.inference_mode()
def predict(net: torch.nn.Module, info: dict, im: Image.Image) -> pd.DataFrame:
    x = preprocess(im, info["preprocess"]).unsqueeze(0)
    probs = torch.sigmoid(net(x))[0].tolist()
    df = pd.DataFrame({"code": info["classes"], "probability": probs, "threshold": info["thresholds"]})
    df["color"] = df["code"].map(NAMES)
    df["predicted"] = df["probability"] >= df["threshold"]
    return df


# ================================================================ page pieces
def verdict(df: pd.DataFrame) -> str:
    hits = df[df["predicted"]].sort_values("probability", ascending=False)
    if hits.empty:
        return "### Colorless\nNo color reached its threshold."
    return "### " + " + ".join(hits["color"]) + f"\n{'Color' if len(hits) == 1 else 'Colors'} above threshold."


def chart(df: pd.DataFrame, theme: str) -> alt.LayerChart:
    """One bar per color (probability), a line across each bar at that color's threshold."""
    df = df.assign(label=df["color"] + "|" + (df["probability"] * 100).round().astype(int).astype(str) + "%",
                   status=df["predicted"].map({True: "predicted", False: "below threshold"}))
    order = df["label"].tolist()  # W U B R G, the fixed color order
    x = alt.X("label:N", sort=order, title=None,
              axis=alt.Axis(labelAngle=0, labelExpr="split(datum.label, '|')", ticks=False))
    tooltip = [alt.Tooltip("color:N", title="Color"),
               alt.Tooltip("probability:Q", title="Probability", format=".1%"),
               alt.Tooltip("threshold:Q", title="Threshold", format=".0%"),
               alt.Tooltip("status:N", title="Result")]
    base = alt.Chart(df).encode(x=x, tooltip=tooltip)
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability",
                axis=alt.Axis(format="%", tickCount=5)),
        color=alt.Color("code:N", legend=None,
                        scale=alt.Scale(domain=list(BAR_COLORS[theme]),
                                        range=list(BAR_COLORS[theme].values()))),
        opacity=alt.condition("datum.predicted", alt.value(1.0), alt.value(0.35)),
    )
    thresholds = base.mark_tick(thickness=2, color=INK[theme]).encode(y="threshold:Q")
    return (bars + thresholds).properties(height=280).configure_scale(bandPaddingInner=0.35)


def footer(info: dict):
    m = info.get("metrics", {})
    st.divider()
    st.caption(
        f"Live model: `{info['run']}` ({info['backbone']}), macro F1 {m.get('macro_f1', float('nan')):.2f} "
        f"on {m.get('n', 0):,} test cards it never saw. "
        f"[Model card](https://huggingface.co/{HF_REPO})\n\n"
        "Unofficial fan project, not affiliated with or endorsed by Wizards of the Coast. "
        "Card art is © Wizards of the Coast and its artists; training data from Scryfall."
    )


# ================================================================ page
def main():
    st.set_page_config(page_title="MTG Color Classifier", page_icon="🎨", layout="centered")
    st.title("Which Magic color is this art?")
    st.write("Upload a picture and a neural network trained on Magic: The Gathering card art "
             "guesses which of the five colors it belongs to: white, blue, black, red, green. "
             "It can pick several colors, or none.")

    net, info = load_model()
    theme = "dark" if st.context.theme.type == "dark" else "light"

    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "bmp", "gif"],
                                max_upload_size=MAX_UPLOAD_MB)
    st.caption("Your image is processed in memory to make the prediction. It is never saved or logged.")

    if uploaded is not None:
        try:
            im = open_upload(uploaded.getvalue())
            im.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            st.error("That file couldn't be read as an image. Try a JPG or PNG.")
        else:
            df = predict(net, info, im)
            left, right = st.columns([1, 1], gap="large")
            with left:
                st.image(im, width="stretch")
            with right:
                st.markdown(verdict(df))
                st.altair_chart(chart(df, theme), width="stretch")
                st.caption("Bar = the model's probability. Line = that color's threshold: "
                           "a color is predicted when its bar reaches the line (solid bars).")
            with st.expander("Numbers"):
                st.dataframe(
                    df[["color", "probability", "threshold", "predicted"]], hide_index=True,
                    column_config={"color": "Color",
                                   "probability": st.column_config.NumberColumn("Probability", format="percent"),
                                   "threshold": st.column_config.NumberColumn("Threshold", format="percent"),
                                   "predicted": "Predicted"})

    footer(info)


if __name__ == "__main__":
    main()
