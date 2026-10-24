# App demo samples

Use these fictional profiles to demonstrate the two possible M1 screening outcomes. They are examples for testing the interface, not real people or medical advice. Complete every question and use the PHP income option as shown.

The scores below are the saved M1 model's internal scores, not the person's probability of having diabetes. M1's decision threshold is **0.3963**.

## Green outcome: screening did not suggest follow-up

| App question | Sample answer |
|---|---|
| Sex | Female |
| Age range | 30–34 |
| Height and weight | 165 cm, 60 kg (BMI rounds to 22) |
| High blood pressure | No |
| High cholesterol | No |
| Stroke | No |
| Coronary heart disease or heart attack | No |
| Fruit at least once a day | Yes |
| Heavy-drinking survey category | No |
| General health | Excellent |
| Poor physical health days in the past 30 days | 0 |
| Poor mental health days in the past 30 days | 0 |
| Serious difficulty walking or climbing stairs | No |
| Annual household income | ₱750,000 (about US$11,826 at the app's configured rate; BRFSS band: US$10,000 to under US$15,000) |

**Expected display:** Green outcome, “This screening did not suggest a follow-up.” Internal M1 score: **0.0401**.

## Yellow outcome: talking with a health professional may be helpful

| App question | Sample answer |
|---|---|
| Sex | Male |
| Age range | 65–69 |
| Height and weight | 165 cm, 90 kg (BMI rounds to 33) |
| High blood pressure | Yes |
| High cholesterol | Yes |
| Stroke | No |
| Coronary heart disease or heart attack | No |
| Fruit at least once a day | No |
| Heavy-drinking survey category | No |
| General health | Fair |
| Poor physical health days in the past 30 days | 10 |
| Poor mental health days in the past 30 days | 5 |
| Serious difficulty walking or climbing stairs | Yes |
| Annual household income | ₱500,000 (about US$7,884 at the app's configured rate; BRFSS band: under US$10,000) |

**Expected display:** Yellow outcome, “It may be worth talking with a health professional.” Internal M1 score: **0.8837**.

## Demo notes

- The income conversion uses ₱63.419 per US$1, the configured BSP reference rate dated 23 October 2026. The app applies the U.S. survey's original income bands without a Philippine cost-of-living adjustment.
- The OpenAI note is optional and may vary between runs. It appears only when OpenAI is configured and the user opts in; the two screening outcomes above do not depend on OpenAI.
- Do not present either fictional profile as evidence about an actual person's health.
