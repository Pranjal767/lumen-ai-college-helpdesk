# AIT fictional demo dataset

The imported corpus in `ait_mu_dataset/` is the ZIP dataset supplied for this project. It contains fictional AIT (Aurora Institute of Technology) scenarios, not verified information about B. K. Birla College or any real institution.

All imported Markdown and CSV records are marked `simulated_demo`, prefixed as demo content in the vector corpus, and indexed with the same label in document metadata. The chatbot must keep that disclosure in answers. Dates, fees, policies, events, and contact details are illustrative. Contact numbers in the corpus are fictional and must never be presented as real college contact information.

To replace the current knowledge corpus and rebuild the local vector index, run `python -m backend.services.ait_demo_dataset` from the project root. Existing conversations are preserved.
