"""Download UCI Bank Marketing (bank-full.csv; CC BY 4.0; Moro et al. 2014) into data/."""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
dest = Path(__file__).resolve().parents[1] / "data"
if (dest / "bank-full.csv").exists():
    raise SystemExit("already downloaded")
dest.mkdir(exist_ok=True)
outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
inner.extract("bank-full.csv", dest)
print("ok", dest / "bank-full.csv")
