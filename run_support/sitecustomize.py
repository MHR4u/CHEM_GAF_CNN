"""
sitecustomize.py  --  capture layer for the gaf_cnn.sh run
==========================================================

Python imports this file automatically at interpreter start when this folder is
on PYTHONPATH. It does NOT change the behaviour of gaf_cnn.py: the training is
exactly the same, only two outputs that would otherwise be lost are captured.

  1) matplotlib: gaf_cnn.py ends with plt.show(). On a headless machine (server,
     no display) plt.show() opens nothing and the training curves would be lost.
     Here the plots are saved as PNG files instead.

  2) training history: model.fit() returns the history object but gaf_cnn.py
     never saves it. It is dumped here to training_history.json so the notebook
     can show the exact curves and per-epoch numbers.

Files are written to $PLOT_OUT_DIR (default: current directory).
"""

import json
import os

OUT_DIR = os.environ.get("PLOT_OUT_DIR", os.getcwd())
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. save matplotlib figures instead of showing them
# --------------------------------------------------------------------------- #
try:
    import matplotlib

    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    _state = {"n": 0}
    _orig_show = plt.show

    def _saving_show(*args, **kwargs):
        """Same as plt.show(), but writes every open figure to a PNG file."""
        _state["n"] += 1
        for num in plt.get_fignums():
            fig = plt.figure(num)
            fig.savefig(os.path.join(OUT_DIR, f"training_curve_{_state['n']}_{num}.png"), dpi=130)
        return None

    plt.show = _saving_show
except Exception as exc:  # pragma: no cover
    print(f"[capture] matplotlib capture disabled: {exc}")


# --------------------------------------------------------------------------- #
# 2. dump the Keras training history (accuracy / loss per epoch)
# --------------------------------------------------------------------------- #
try:
    import keras

    _keras_model = keras.Model
    _orig_fit = _keras_model.fit

    def _saving_fit(self, *args, **kwargs):
        history = _orig_fit(self, *args, **kwargs)
        try:
            data = {k: [float(x) for x in v] for k, v in history.history.items()}
            with open(os.path.join(OUT_DIR, "training_history.json"), "w") as f:
                json.dump(data, f, indent=2)
            print(f"[capture] training history saved -> {os.path.join(OUT_DIR, 'training_history.json')}")
        except Exception as exc:  # pragma: no cover
            print(f"[capture] could not save training history: {exc}")
        return history

    _keras_model.fit = _saving_fit
except Exception as exc:  # pragma: no cover
    print(f"[capture] training history capture disabled: {exc}")
