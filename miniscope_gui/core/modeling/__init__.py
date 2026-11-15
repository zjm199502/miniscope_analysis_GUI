"""Step 9: mathematical models linking population activity and behaviour.

Everything here is plain numpy/scipy so the app stays small: no scikit-learn,
no torch. Each analysis has a ``run_*`` function with the usual worker
signature ``(session, ws, params, progress, should_stop)`` that stores its
result in ``modeling/<key>.npz`` and returns a short summary for the GUI.
"""
