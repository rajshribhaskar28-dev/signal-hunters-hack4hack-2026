# Submission checklist — Round 2 (closes 27 Oct 2026, 11:59 PM IST)

1. **Upload predictions.** Team workspace → **Leaderboard** tab (Round 2) → upload `submission_final.csv`.
   - Up to 5 uploads are allowed.
   - A file that fails the format checks is rejected and doesn't use an upload.
   - `submission_v1.csv` is a weaker fallback; upload it only if you want a comparison point.
2. **Choose the scoring upload.** The final ranking uses each team's *best public* upload, scored on the hidden 70%. If you upload both files, check which one counts.
3. **Publish the code.** Create a GitHub repository, for example `signal-hunters-r2`:
   - Unzip `signal_hunters_code.zip` into it.
   - `git add . && git commit -m "Round 2 solution" && git push`.
   - Don't commit the competition data. It isn't included in the zip.
4. **Submit the link.** Team workspace → **Submission** tab → paste the repository URL **before 11:59 PM IST**.
5. **Reproduction command** (in the README):
   `python train_pipeline.py --data <folder with the competition files> --out submission.csv`
