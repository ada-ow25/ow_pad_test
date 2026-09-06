# CPU-only by design: the shipped model is a logistic regression over 9
# hand-computed features, so there is nothing to gain from a GPU and a lot to
# lose in build complexity and portability.
FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless still needs libGL and glib at runtime even though it
# draws nothing -- installing them here rather than debugging an ImportError
# at container start. --no-install-recommends keeps the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, in their own layer, so editing source doesn't trigger a
# reinstall on rebuild. Pinned to the versions the analysis was run with.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what inference needs: the feature formulas, the CLI, and the fitted
# artefacts. Notebooks, tests and training data stay out of the image.
COPY features.py infer.py ./
COPY artefacts/ ./artefacts/

# Fail fast at build time if the artefacts didn't make it in, rather than at
# run time on the grader's machine.
RUN python -c "import json, joblib; joblib.load('artefacts/model.joblib'); \
    json.load(open('artefacts/config.json')); print('artefacts load OK')"

# Mount a folder of images at /data and the results land in /out:
#   docker build -t pad .
#   docker run --rm -v /path/to/images:/data -v $(pwd):/out pad
ENTRYPOINT ["python", "infer.py"]
CMD ["--input", "/data", "--output", "/out/scores.csv"]
