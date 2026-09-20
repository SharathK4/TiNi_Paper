"""Version 4 — Inflect: generic curve analysis for uploaded experimental data.

Versions 1–3 are one paper, one dataset, one model. This package is the part
that generalises: given *any* table of numbers from an experiment or a
simulation, work out what the columns mean, where the curve changes regime,
and what each regime is worth — without the user writing code or choosing a
single parameter by hand.

    ingest      read a file whatever its dialect; name its columns
    columns     decide which column is the independent variable and which
                signals are worth plotting, and say why
    noise       estimate the noise level and pick a smoothing width from it
    segment     split a curve into regimes (constant / linear / quadratic)
                separated by jumps and knees — the core of the product
    transitions turn segment boundaries into physical transition points with
                uncertainty, and derive what each regime is worth
    ensemble    the same analysis over many files: statistics and outliers
    fit         user-chosen models on user-chosen ranges
    sessions    an upload plus its analysis, stored and resumable
"""
