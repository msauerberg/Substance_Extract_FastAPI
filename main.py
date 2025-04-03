from fastapi import FastAPI, UploadFile, File
import pandas as pd
from io import StringIO
from fastapi.responses import StreamingResponse
from StringMatching import get_substances

app = FastAPI()


def create_output(substance_data, ref_data):
    """Process CSV data using get_substances function"""

    # Ensure required columns exist
    if "Substance" not in substance_data.columns or "Substanz" not in ref_data.columns:
        raise ValueError(
            "Missing required columns: 'Substance' in file1 or 'Substanz' in file2."
        )

    input_col = substance_data["Substance"]
    ref_subs = ref_data["Substanz"]

    out = get_substances(reference_series=ref_subs, input_col=input_col)

    return out


@app.post("/process/")
async def upload_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):

    df1 = pd.read_csv(file1.file, sep=None, engine="python")
    df2 = pd.read_csv(file2.file, sep=None, engine="python")
    the_output = create_output(df1, df2)
    output_csv = StringIO()
    the_output.to_csv(output_csv, index=False)
    output_csv.seek(0)

    return StreamingResponse(
        output_csv,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=processed.csv"},
    )
