# Bank Marketing deciles

CatBoost was selected by validation PR-AUC for the cleaned arm at seed 42. The frozen classification threshold was 0.219666. Source run `bank_marketing_20261112-172256` used protocol `55568f7e456ee7e122df427f87230b27037b3e5de5250f24ae04c85e36933c29`.

- **Train (in-sample)**. Contacting the highest-scored 30% of customers would reach 80.7% of subscribers in this split.
- **Validation**. Contacting the highest-scored 30% of customers would reach 69.8% of subscribers in this split.
- **Test**. Contacting the highest-scored 30% of customers would reach 72.0% of subscribers in this split.

## Held-out response-rate checks

- Validation rose from 3.54% in decile 8 to 4.09% in decile 9.
- Test rose from 3.54% in decile 7 to 3.87% in decile 8.
- Test rose from 3.87% in decile 8 to 3.98% in decile 9.

Customers with equal scores are ranked by source-row ID. The train split is in-sample.

| Split | Customers | Subscribers | Table | Chart |
| --- | ---: | ---: | --- | --- |
| Train (in-sample) | 27,126 | 3,173 | [CSV](decile_bank_marketing_train.csv) | [Chart](decile_bank_marketing_train.png) |
| Validation | 9,042 | 1,058 | [CSV](decile_bank_marketing_validation.csv) | [Chart](decile_bank_marketing_validation.png) |
| Test | 9,043 | 1,058 | [CSV](decile_bank_marketing_test.csv) | [Chart](decile_bank_marketing_test.png) |
