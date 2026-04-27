"""
polyglot_talk — Real-time offline Speech-to-Speech Translation.

Core pipeline package. Import order matters: config sets OS env vars
(OMP_NUM_THREADS, CT2_INTER_THREADS) before any CTranslate2 library loads.
"""
