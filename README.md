# March Machine Learning Mania 2026 — 1st Place Solution

**Final Score: 0.1097454 (MSE/Brier, 126 games) | Rank: 1st / 3,485 teams**

### Approach
Logistic Regression on team-level differential features + triple-market probability blend (ESPN BPI, Vegas moneylines, Kalshi prediction markets). Separate men's/women's models. 52 rounds of iteration.

See SOLUTION_WRITEUP.md for the full write-up.

### Reproduction
Requirements
Python 3.9+
numpy, pandas, scikit-learn, scipy
The pipeline also requires official Kaggle competition data:

pip install kaggle
kaggle competitions download -c march-machine-learning-mania-2026 -p data/
cd data && unzip '*.zip' && rm *.zip && cd ..
Generate Submission
python3 -c "
import round52_final as r52
sub = r52.generate_submission(alpha_m_r1=0.90, alpha_w_r1=0.75)
sub.to_csv('submission.csv', index=False)
print(f'Wrote {len(sub)} rows to submission.csv')
"
Key parameters for the winning submission:

File Structure
external/           — External datasets (Barttorvik, KenPom, EvanMiya, AP Poll, etc.)
Import chain: round27 → round45 → round46 → round49 → round50 → round51 → round52

## Competition
March Machine Learning Mania 2026 on Kaggle.
