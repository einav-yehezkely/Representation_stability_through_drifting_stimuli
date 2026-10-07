# תוכנית בנייה מפורטת — ניסוי אנושי (Human Experiment)

מסמך זה הוא תוכנית עבודה לבניית האתר כפי שמוגדר ב-[HUMAN_EXPERIMENT_PRD.md](HUMAN_EXPERIMENT_PRD.md). התוכנית לא מוסיפה דרישות; כל פרמטר מדעי בה נגזר מקוד המקור שנבדק (סעיף 1), וכל מקום שבו המקור סותר את עצמו או עמום מופיע בסעיף 2 כשאלה פתוחה, ולא כהחלטה.

---

## 0. כללי הגנה על הריפו (PRD §2) — חלים על כל שלב

1. **לפני תחילת העבודה**: לשמור צילום מצב (baseline) מחוץ לריפו (בתיקיית scratchpad):
   - `git status --porcelain=v1 -uall`
   - `git diff` מלא (כרגע `human_experiment/HUMAN_EXPERIMENT_PRD.md` כבר מסומן M, וה-PDF הוא untracked; שניהם חייבים להישאר כמו שהם)
   - `git ls-files -s` (hash של כל קובץ במעקב) ו-SHA-256 של `pca_top2_filtered_female_vgg_1.csv`
2. כל קובץ חדש נוצר **רק** תחת `human_experiment/`. לא נוגעים ב-`.gitignore` שבשורש, ב-`.vscode/`, ב-`.venv/` או בכל קובץ קיים, וגם לא ב-`HUMAN_EXPERIMENT_PRD.md` עצמו.
3. `npm install`, `npm run build` וכל הסקריפטים רצים עם `cwd = human_experiment/`. הפלט שלהם (`node_modules/`, `dist/`, `public/assets/`) נשאר בתוך התיקייה. נוצר `human_experiment/.gitignore` משלנו.
4. `female_faces/` ו-`pca_top2_filtered_female_vgg_1.csv` נקראים **לקריאה בלבד** על ידי סקריפט ההכנה. העותקים והקבצים הנגזרים נכתבים רק ל-`human_experiment/public/assets/`.
5. קוד Python מקורי שנדרש לבדיקת התאמה (parity) **מועתק** ל-`human_experiment/scripts/reference/`, ורק העותק משמש.
6. **בסוף העבודה**: מריצים שוב את פקודות ה-baseline ומשווים. השינויים היחידים המותרים הם קבצים חדשים תחת `human_experiment/`. התוצאה מתועדת ב-README (בדיקת קבלה 1).

---

## 1. ממצאי בדיקת המקור (PRD §3) — הבסיס המדעי

| נושא | מקור | מה נמצא |
| --- | --- | --- |
| תמונות | `female_faces/` | שמות קבצים כמו `000004.jpg`, JPEG בגודל 178×218. מתוך 11,817 התמונות שב-PCA, לכולן קיים קובץ (סה"כ ‎85.7MB). |
| נתוני PCA | `pca_top2_filtered_female_vgg_1.csv` | ללא כותרת, עמודות `filename, PC1(x), PC2(y)`. 11,817 שורות, ללא שמות כפולים. נטען ב-`load_top2_filtered()` ([classify_rotation_resnet50.py:236](../network_classification/classify_rotation_resnet50.py#L236)). הקובץ נוצר ב-[PCA_2_and_rest.py](../extract_embeddings/PCA_2_and_rest.py) (InceptionResNetV1, 10% השאריות הקטנות) וסונן ב-[filter_pca.py](../network_classification/tools/filter_pca.py). |
| מוסכמת זווית | [classify_rotation_resnet50.py:52-56](../network_classification/classify_rotation_resnet50.py#L52-L56) | `angle_deg = degrees(atan2(y, x)) % 360`, ראשית ב-(0,0). המוסכמה **נתמכת במקור**, ולכן מותר להשתמש בה. |
| מרכז A ההתחלתי | `create_base_and_opposite_points(angle)` ([:255](../network_classification/classify_rotation_resnet50.py#L255)) | `target_radius = 0.45`. נבחרת **נקודת דאטה אמיתית** שממזערת `angle_error + 100·|r − 0.45|`. ל-0° זו `183785.jpg` = (0.44447, ‎−0.00393), זווית ‎359.49°, רדיוס 0.4445. |
| מרכז B | אותה פונקציה | `opposite_point = -base_point` (שיקוף מדויק). |
| סיבוב | `rotate_vector` ([:300](../network_classification/classify_rotation_resnet50.py#L300)) | מטריצת סיבוב **נגד כיוון השעון** (חיובי = CCW). העדכון **אינקרמנטלי**: אחרי כל איטרציה `base_point = rotate_vector(base_point, ROTATION_DEGS)` ([:1783-1784](../network_classification/classify_rotation_resnet50.py#L1783-L1784)), אחרי הבחירה והאימון, ולא לפניהם. |
| בניית אשכול | `collect_nearest_images` ([:321](../network_classification/classify_rotation_resnet50.py#L321)) | `k` הנקודות הקרובות ביותר במרחק אוקלידי דו-ממדי למרכז הנוכחי, ממוינות לפי מרחק (`argpartition` ואחריו `argsort`). האשכול מחושב מחדש בכל איטרציה. **אין** החרגה של תמונות שכבר הוצגו בין איטרציות. |
| גודל אשכול | `NUM_OF_IMAGES_PER_CLUSTER` ([:1467](../network_classification/classify_rotation_resnet50.py#L1467)) | בקוד 64; ב-commit ‏350e7ce נקבע 65; בסמינר (§3.4) כתוב 100 → הוכרע: k=1 (ה-1). |
| בחירת גירוי | לולאת `__main__` | הרשת מקבלת **את כל** 2k התמונות בכל איטרציה, באותו משקל. במקור אין שלב "בחירת פנים אחת". |
| אימון ראשוני של הרשת | [generate_rotation_sequence.py:363-381](../network_classification/tools/generate_rotation_sequence.py#L363-L381) ו-`resnet50_embeddings_training.py` (`split_data`) | k=1000 סביב הנקודה **המדויקת** (0.45·cosθ, 0.45·sinθ) ולא סביב נקודת דאטה, ובקובץ PCA אחר (`..._20-30percent.csv`) → לא בשימוש; באימון: nearest-unused (ה-4). |
| רצף טרג'קטוריה | `generate_rotation_sequence(... num_steps=360, rotation_range=360, used_indices=set())` ([:1491](../network_classification/classify_rotation_resnet50.py#L1491)) | 360 תמונות ייחודיות, הקרובות ביותר לנקודת הבסיס המסובבת בכל מעלה. משמש להערכה ולגרפים. |
| גרפים | `create_prediction_scatter` ([:807](../network_classification/classify_rotation_resnet50.py#L807)), `plot_clusters_with_given_indices` ([generate_rotation_sequence.py:205](../network_classification/tools/generate_rotation_sequence.py#L205)) | כל הנקודות באפור (s=5, alpha=0.3); A כחול, B אדום; מרכזים כ-× שחור (בסיס) ו-* (נגדי); מעגל מקווקו ברדיוס `max(r)·1.05`; קווים רדיאליים ותוויות כל 20°; צירים x=0/y=0; grid; `axis equal`; תוויות PC1/PC2. |
| אקראיות | [:25-29](../network_classification/classify_rotation_resnet50.py#L25-L29) | `SEED = 42`. ב-[merge_sequences.py](../network_classification/tools/merge_sequences.py) A/B מעורבבים בהסתברות 0.5 (לא בלוקים מאוזנים). |

---

## 2. החלטות החוקרת ושאלות שנותרו

### 2.1 החלטות שהתקבלו (מחייבות)

| # | נושא | החלטה | מימוש |
| --- | --- | --- | --- |
| ה-1 | גודל אשכול | **k = 1** | `scientific.clusterSize: 1`. במקום ה-64 של המקור משתמשים באותה פונקציה, `collect_nearest_images`, עם k=1 |
| ה-2 | בחירת פנים אחת | **אשכול בגודל 1 → התמונה היחידה בו** | הבחירה **דטרמיניסטית**: התמונה הקרובה ביותר (מרחק אוקלידי דו-ממדי) למרכז של הקבוצה שנבחרה. אין אקראיות בבחירת התמונה; האקראיות היחידה היא בסדר הקבוצות |
| ה-3 | מרכז B | **תמיד 180° ממרכז A** | `B = −A` (נקודת דאטה של A משוקפת דרך הראשית, כמו `opposite_point = -base_point` במקור). שני הפרמטרים `initialAngleA/B` נשארים בקונפיגורציה (PRD §6), אבל `validateConfig` **דוחה** כל ערך שבו `(initialAngleB − initialAngleA) mod 360 ≠ 180`. הזווית הנומינלית של B נלקחת מהקונפיג, והמיקום בפועל הוא `−A` |
| ה-4 | בחירה בשלב האימון | **בכל ניסיון, התמונה הקרובה ביותר למרכז הקבוצה שעדיין לא נבחרה** | קבוצת `usedTrainingIndices` (כמו `used_indices` ב-`generate_rotation_sequence` במקור). בכל ניסיון אימון: nearest מתוך התמונות שלא ב-`usedTrainingIndices`. התמונה נכנסת לקבוצה **רק אחרי תגובה שנשמרה** (בכשל טעינה מנסים שוב את אותה תמונה) |
| ה-5 | חזרות בסחף | **תמונת סחף לא חוזרת בתוך הסחף; מותר להשתמש בסחף בתמונות מהאימון** | קבוצה נפרדת `usedDriftIndices`, שמתחילה ריקה בתחילת הסחף. בכל ניסיון סחף: nearest מתוך התמונות שלא ב-`usedDriftIndices`. תמונות האימון **לא** מוחרגות בסחף |
| ה-6 | קובץ PCA | **`pca_top2_filtered_female_vgg_1.csv`** | ללא שינוי. `female_facenet_vggface2_embeddings.csv` (512 ממדים, 118,164 תמונות) הוא קובץ ה-embeddings שממנו ה-PCA נגזר ([PCA_2_and_rest.py](../extract_embeddings/PCA_2_and_rest.py)), ואינו בשימוש באתר |

### 2.2 השלכות שנמדדו על הנתונים האמיתיים (סימולציה ב-Node, ברירות מחדל 0°/180°, צעד 1°)

- מרכז A = `183785.jpg` (‎359.49°, r=0.4445); לכן ניסיון האימון הראשון מקבוצה A יציג את התמונה הזו עצמה (מרחק 0). מרכז B נמצא ב-‎179.49°, והתמונה הקרובה אליו במרחק 0.0017.
- **אימון**: המרחק למרכז גדל ככל שהתמונות הקרובות מתמצות. ב-A: 0.037 בניסיון ה-20 ו-0.060 בניסיון ה-60 של הקבוצה. ב-B: 0.018 ו-0.032. המרכזים עצמם נשארים קבועים, אבל התמונות המוצגות מתרחקות מהם בהדרגה.
- **סחף בלי חזרות (ה-5)**: מרכזי A ו-B עוברים על אותה טבעת. לכן כשמרכז B מגיע לאזור ש-A כבר עבר בו, התמונות הקרובות ביותר שם כבר נוצלו, ונבחרת התמונה הבאה בתור. בסימולציה עם סדר קבוצות אקראי ומאוזן (180/180), המרחק בין התמונה למרכז היה: חציון 0.006, אחוזון 90 ‏0.058, מקסימום 0.104 (רדיוס המרכז 0.445).
- הסחף לא תלוי במספר ניסיונות האימון, כי תמונות האימון אינן מוחרגות בו. לכן רצף תמונות הסחף נקבע רק לפי סדר הקבוצות בסחף.
- המרחק בפועל נשמר בכל רשומה (`distanceToCenter`) לצורך הניתוח.

### 2.3 שאלות שעדיין פתוחות

| # | שאלה | ברירת מחדל עד החלטה |
| --- | --- | --- |
| ש-ב | זווית "מוצגת" לעומת זווית אמיתית: המרכז ב-0° נמצא בפועל ב-‎359.49° | `currentAngleA/B` = הזווית הנומינלית (`initial + (k−1)·step`, כמו בטבלת ה-PRD), ובנוסף עמודות `centerA_pc1/pc2/actualAngle` |
| ש-ג | "trajectory membership" לוויזואליזציה | שכבה ראשית: התמונות הקרובות ביותר למרכזי A ו-B בכל מיקומי הסחף (לפי הקונפיג האפקטיבי, בלי החרגה). שכבה נוספת שאפשר להפעיל: רצף `generate_rotation_sequence` המקורי |
| ש-ד | פרמטרי UI שאינם ב-PRD | placeholders מסומנים: `feedbackDurationMs: 1000`, `trainingBlockSize: 4`, תמונה במידות המקור (178×218) מוגדלת ×2 |

---

## 3. סטאק טכנולוגי

- **Vite + JavaScript ES modules (vanilla)**, בלי framework. האפליקציה קטנה וה-PRD מבקש "practical, no unnecessary infrastructure".
- **Vitest + jsdom** לבדיקות יחידה ובדיקות אינטגרציה של המנוע וה-UI. **Playwright** (אופציונלי, devDependency בתוך `human_experiment/`) לבדיקת smoke בדפדפן אמיתי: תזמון onset, חלון הורדה.
- **Node scripts** להכנת נכסים. Python נדרש רק לסקריפט parity, בתוך venv מבודד ב-`human_experiment/.venv-parity/` (ה-`.venv` שבשורש לא כולל pandas ולא ייגעו בו).
- שני דפים (Vite multi-page): `index.html` לניסוי, `review.html` לתצוגת החוקרת.

---

## 4. מבנה קבצים (לפי PRD §16)

```text
human_experiment/
├── .gitignore                     # node_modules, dist, public/assets/faces, .venv-parity
├── README.md
├── IMPLEMENTATION_PLAN.md         # המסמך הזה
├── package.json / package-lock.json
├── vite.config.js                 # multi-page, outDir=dist (בתוך התיקייה)
├── index.html                     # ניסוי (participant/dev)
├── review.html                    # תצוגת חוקרת (ויזואליזציה)
├── public/assets/
│   ├── faces/                     # עותקים של התמונות (נוצר בסקריפט, gitignored)
│   ├── pca.json                   # נגזר מה-CSV: [{imageId, pc1, pc2}] + metadata
│   └── assets-manifest.json       # sha256 של ה-CSV, מספר תמונות, hash של רשימת התמונות
├── scripts/
│   ├── prepare-assets.mjs         # read-only על המקור → public/assets
│   ├── verify-repo-unchanged.mjs  # השוואה ל-baseline
│   └── reference/
│       ├── source_copy.py         # עותק מילולי של הפונקציות המדעיות מהמקור
│       └── make_parity_fixtures.py# מייצר tests/fixtures/parity/*.json
├── src/
│   ├── main.js                    # composition root: config → storage adapter → engine → UI
│   ├── review.js                  # composition root של תצוגת החוקרת
│   ├── config/
│   │   ├── experimentConfig.js    # נקודת כניסה יחידה לכל הפרמטרים
│   │   └── validateConfig.js
│   ├── content/
│   │   ├── consent.js             # טקסט עם [RESEARCHER: ...] placeholders
│   │   ├── instructions.js
│   │   └── uiText.js              # transition, feedback, errors, completion
│   ├── components/
│   │   ├── consentScreen.js
│   │   ├── instructionsScreen.js
│   │   ├── trialScreen.js
│   │   ├── transitionScreen.js
│   │   ├── completionScreen.js
│   │   ├── recoveryScreen.js
│   │   ├── errorBanner.js
│   │   └── devPanel.js            # נטען רק במצב פיתוח
│   ├── experiment/
│   │   ├── experimentEngine.js    # state machine
│   │   ├── trainingPhase.js       # קריטריון חלון נע
│   │   ├── driftPhase.js          # סדר תנועה מדויק
│   │   ├── responseCoding.js
│   │   ├── seededRandomization.js
│   │   ├── stimulusPresenter.js   # טעינה/decode/onset (DOM-aware, מוזרק)
│   │   └── session.js             # sessionId, participantId, metadata
│   ├── scientific/                # ללא תלות ב-DOM/storage
│   │   ├── pcaLoader.js
│   │   ├── trajectory.js          # angleDeg, rotateVector, createBasePoint
│   │   ├── clusterConstruction.js # collectNearest(k)
│   │   └── stimulusSelector.js
│   ├── visualization/
│   │   ├── pcaPlot.js             # SVG renderer (מקביל ל-matplotlib במקור)
│   │   └── plotExport.js          # SVG + PNG
│   └── services/dataStorage/
│       ├── storageContract.js
│       ├── index.js               # createStorage(config) → adapter
│       ├── localCsvStorage.js
│       └── csvSerializer.js
└── tests/
    ├── unit/ …                    # לפי מודול
    ├── integration/ …             # engine + fake presenter + memory storage
    ├── fixtures/parity/*.json
    └── e2e/ (Playwright, אופציונלי)
```

---

## 5. שלבי בנייה

כל שלב מסתיים בבדיקות ירוקות לפני שעוברים לשלב הבא.

### שלב 0 — Baseline והקמה
1. שמירת baseline לפי סעיף 0.1.
2. `npm create vite` ידני (כתיבת `package.json` בלבד, בלי scaffold שכותב לשורש), התקנת `vite`, `vitest`, `jsdom` ו-(`@playwright/test`).
3. `.gitignore` מקומי, `vite.config.js` עם `root: human_experiment`, `build.outDir: 'dist'`, ו-`input: {index, review}`.
4. סקריפטי npm: `prepare-assets`, `dev`, `build`, `preview`, `test`, `test:e2e`, `parity:fixtures`, `verify:repo`.

### שלב 1 — הכנת נכסים (`scripts/prepare-assets.mjs`)
1. קורא את `../pca_top2_filtered_female_vgg_1.csv` (ללא כותרת) ושומר את מחרוזות המספרים המקוריות. ההמרה נעשית עם `Number()` (float64, כמו numpy; הבדל אפשרי של ulp אחד מול parser של pandas ייבדק ב-parity).
2. בודק התאמה: 3 עמודות, שמות לא כפולים, ערכים סופיים, וקיום קובץ לכל שם ב-`../female_faces/`. אם משהו חסר, הסקריפט נכשל עם רשימת הקבצים החסרים ולא מדלג עליהם בשקט.
3. מעתיק את 11,817 התמונות ל-`public/assets/faces/`. מעתיקים את כל המאגר ולא רק את האשכולות, כי `degreesPerTrial` ו-k ניתנים להגדרה.
4. כותב את `pca.json` ואת `assets-manifest.json` (sha256 של ה-CSV, מספר שורות, hash של רשימת התמונות, תאריך יצירה). מחרוזות הגרסה האלה נכנסות ל-metadata של הסשן.

### שלב 2 — קונפיגורציה מרכזית (`src/config/experimentConfig.js`)
```js
export const experimentConfig = {
  experimentVersion: "1.0.0",
  schemaVersion: "1",
  training: { initialAngleA: 0, initialAngleB: 180, minTrials: 20, accuracyWindow: 20, accuracyThreshold: 0.80 },
  drift:    { numberOfTrials: 360, degreesPerTrial: 1 },
  assets:   { facesDirectory: "assets/faces/", pcaDataPath: "assets/pca.json", manifestPath: "assets/assets-manifest.json" },
  scientific: {
    // נגזר מ-network_classification/classify_rotation_resnet50.py
    origin: [0, 0],                    // atan2 around origin (source line 52-56)
    angleConvention: "atan2(PC2,PC1) mod 360, degrees",
    rotationDirection: "counterclockwise",   // rotate_vector, source line 300
    targetRadius: 0.45,                // create_base_and_opposite_points, line 276
    radiusErrorWeight: 100,            // combined_error = angle_error + radius_error*100, line 288
    clusterSize: 1,                    // researcher decision ה-1 (source NUM_OF_IMAGES_PER_CLUSTER=64)
    clusterDistance: "euclidean2D",    // collect_nearest_images
    centerB: "negateA",                // researcher decision ה-3: B = −A (source opposite_point = -base_point)
    trainingSelection: "nearestUnused",// researcher decision ה-4 (source used_indices pattern)
    driftSelection: "nearestUnused",   // researcher decision ה-5: no repeats in drift
    exclusionScope: "perPhase",        // separate used sets: training excludes training, drift excludes drift only (ה-4, ה-5)
  },
  randomization: { seedPolicy: "crypto-random-uint32", algorithm: "sfc32", algorithmVersion: "1", trainingBlockSize: 4 /* placeholder ש-ד */ },
  ui: { feedbackDurationMs: 1000, imageDisplayScale: 2 },
  storage: { adapter: "localCsv", localStoragePrefix: "hx" },
  development: { enabled: false, allowUrlActivation: false },
};
```
- `validateConfig()` רץ לפני ה-consent. הוא בודק: זוויות סופיות; `(initialAngleB − initialAngleA) mod 360 === 180` (ה-3); ספירות שהן מספרים שלמים חיוביים; `accuracyWindow ≤ minTrials` אינו נדרש (התנאי `completed ≥ max(minTrials, window)` מטופל בקוד); סף ב-[0,1]; `trainingBlockSize` זוגי; `clusterSize ≤ N`; ערכי enum נתמכים; וטעינה מוצלחת של נתיבי הנכסים. בכישלון מוצגת שגיאה באנגלית והניסוי לא מתחיל.
- `resolveEffectiveConfig(base, devOverrides)` מחזיר עותק עמוק קפוא, וכל override נרשם ב-`metadata.developmentOverrides`.
- אין זוויות התחלה כפולות: `drift` לא מחזיק זוויות משלו ומשתמש ב-`training.initialAngleA/B`.

### שלב 3 — מודולים מדעיים (טהורים, ללא DOM)
| מודול | פונקציה | מקביל במקור |
| --- | --- | --- |
| `trajectory.js` | `angleDeg([x,y]) = ((atan2(y,x)·180/π) % 360 + 360) % 360` | ANGLE_MAP |
| | `rotateVector(v, deg)`: מטריצה CCW, אותו סדר פעולות כמו numpy | `rotate_vector` |
| | `createBasePoint(points, angle, {targetRadius, radiusErrorWeight})` → `{index, point}` עם argmin, ובשוויון האינדקס הראשון (כמו `np.argmin`) | `create_base_and_opposite_points` |
| | `initialCenters(cfg, pca)`: A = base point; B = `−A` (ה-3) | `opposite_point = -base_point` |
| `clusterConstruction.js` | `collectNearest(center, points, k, excluded?)` → אינדקסים ממוינים לפי מרחק אוקלידי, ובשוויון לפי אינדקס (מתועד; ב-numpy הסדר בשוויון לא מוגדר). אינדקסים ב-`excluded` מקבלים מרחק ∞, כמו `dists[idx] = np.inf` במקור | `collect_nearest_images` + `used_indices` של `generate_rotation_sequence` |
| `stimulusSelector.js` | `selectStimulus({group, centers, pca, excluded, cfg})` → `{imageIndex, imageId, pc1, pc2, stimulusAngle, distanceToCenter}`. אימון: `collectNearest(center, k=1, usedTrainingIndices)`; סחף: `collectNearest(center, k=1, usedDriftIndices)`. פונקציה טהורה, בלי RNG. אם כל התמונות נוצלו, נזרקת שגיאה מפורשת (לא יקרה בברירות המחדל: 11,817 תמונות) | ה-1, ה-2, ה-4, ה-5 |
| `pcaLoader.js` | טעינת `pca.json` ו-manifest, ולידציה, החזרת `Float64Array`s | `load_top2_filtered` |

- מרכזים מוחזקים כ**וקטורים** ומסובבים **אינקרמנטלית** אחרי כל תגובה, כמו במקור, ולא בנוסחה סגורה. הזווית הנומינלית מחושבת בנפרד לתיעוד.

### שלב 4 — אקראיות זרועה (`seededRandomization.js`)
- `sessionSeed` = uint32 מ-`crypto.getRandomValues` (יצירת ה-seed עצמה מתועדת), או seed שנקבע במצב פיתוח.
- PRNG אחד מתועד (sfc32), ושני **sub-streams** נגזרים מאותו seed בעזרת hash קבוע: `trainingSchedule` ו-`driftSchedule`. כך מספר ניסיונות האימון, שתלוי בתגובות, לא מזיז את רצף הסחף. אין stream לבחירת תמונה, כי הבחירה דטרמיניסטית (ה-2, ה-4). בקוד הניסוי אין `Math.random()`, ויש בדיקת lint/grep שמוודאת זאת.
- **סחף**: רשימה עם `floor(N/2)` A ו-`floor(N/2)` B; אם N אי-זוגי, הקבוצה העודפת נבחרת ב-RNG. הרשימה מעורבבת ב-Fisher–Yates ונוצרת מראש. ההפרש בין הקבוצות ≤ 1.
- **אימון**: בלוקים של `trainingBlockSize` (חצי A, חצי B), כל בלוק מעורבב ב-Fisher–Yates. בלוק חדש נוצר רק כשהקודם נגמר.
- ב-metadata נשמרים: seed, algorithm+version, ה-schedule האפקטיבי של הסחף, בלוקי האימון שנוצרו, ומונה draws לכל stream. יחד עם התגובות ונתוני ה-PCA, זה מספיק לשחזור מלא של רצף התמונות (כולל `usedTrainingIndices` ו-`usedDriftIndices`, שנגזרים מהרשומות). מדיניות האיזון מתועדת ב-README.

### שלב 5 — קידוד תגובה וקריטריון אימון
- `responseCoding.js`: `RESPONSES = ["A","Probably A","Probably B","B"]`; `toBinary()`; `isCorrect(binary, trueGroup)`.
- `trainingPhase.js`:
  ```js
  evaluateCriterion(history, {minTrials, accuracyWindow, accuracyThreshold}) {
    const n = history.length;
    const eligible = n >= minTrials && n >= accuracyWindow;
    if (!eligible) return { eligible, accuracy: null, passed: false };
    const last = history.slice(-accuracyWindow);
    const accuracy = last.filter(t => t.correct).length / accuracyWindow;
    return { eligible, accuracy, passed: accuracy >= accuracyThreshold };
  }
  ```
  הקריטריון נבדק אחרי **כל** תגובה. אין cap ואין timeout. הערך נשמר ברשומה (`windowAccuracy` כשדה נוסף).

### שלב 6 — מנוע הניסוי (`experimentEngine.js`)
State machine:
`init → (recovery?) → consent → instructions → training ⇄ feedback → transition → drift → completing → complete` (+ `error`, `devTerminated`).

**לולאת ניסיון (משותפת לשני השלבים, לפי הסדר המדויק ב-PRD §7):**
1. `snapshot = {centerA, centerB, nominalAngleA, nominalAngleB}` — העתקה, לא הפניה.
2. `group = schedule.next()`.
3. `stim = selectStimulus(group, snapshot)`: nearest מתוך התמונות שעדיין לא הוצגו **באותו שלב** (ה-4, ה-5).
4. `await presenter.show(stim.imageId)` → מחזיר `onset` (ראו שלב 7). **בכישלון טעינה**: מוצגת שגיאה באנגלית עם כפתור Retry; ה-schedule **לא** מתקדם (ה-group וה-stim נשמרים לניסיון חוזר), אין רשומה ואין סיבוב.
5. תגובה ראשונה מתקבלת → `rt = performance.now() − onset` (נלכד **ראשון**), הכפתורים מושבתים מיד ונוצרת רשומה עם ה-snapshot. `await storage.saveTrial(record)`; אם השמירה נכשלה, מוצגת שגיאה גלויה ואין התקדמות. רק אחרי שמירה מוצלחת, התמונה נוספת לקבוצת ה-used של השלב הנוכחי (`usedTrainingIndices` או `usedDriftIndices`).
6. **אימון**: פידבק `Correct`/`Incorrect` למשך `feedbackDurationMs`, ואחריו הערכת קריטריון (על ניסיון שכבר קיבל פידבק). אם עבר → `transition`.
   **סחף**: `centerA = rotateVector(centerA, step)`, `centerB = rotateVector(centerB, step)`, `driftIndex++`.
7. ניסיון הבא, או `completing` אחרי `numberOfTrials` תגובות סחף.

- ב-`transition` הסחף מתחיל רק אחרי לחיצה על `Continue`.
- הגנה מכפולות: דגל `acceptingResponse` ברמת המנוע ו-`disabled` ברמת ה-DOM. מפתח הרשומה הוא `${sessionId}:${trialNumber}`, ו-`saveTrial` אידמפוטנטי (מפתח קיים → לא נכתב שוב).
- המנוע לא יודע על DOM ולא על localStorage; הוא מקבל `presenter`, `storage`, `clock` ו-`rngFactory` בהזרקה. זה מאפשר בדיקות עם fakes.

### שלב 7 — הצגת תמונה ותזמון (`stimulusPresenter.js`)
נוהל onset מתועד:
1. `const img = new Image(); img.src = url; await img.decode()`. טעינה בלבד, הטיימר עוד לא רץ.
2. החלפת התמונה הקודמת ב-DOM (התמונה הקודמת מוסתרת כבר אחרי התגובה, כדי שלא תוצג פנים ישנה).
3. `requestAnimationFrame(() => requestAnimationFrame(t => { onset = performance.now(); enableButtons(); }))`: ה-rAF השני רץ אחרי שהפריים עם התמונה צויר. ה-onset וה-enable קורים באותו רגע.
4. Preload של הניסיון הבא: מותר רק אחרי שהבחירה שלו נקבעה. בסחף אפשר רק אחרי הסיבוב; באימון אפשר רק אחרי הערכת הקריטריון. preload לא משנה את סמנטיקת ה-onset.
5. `onerror`/`decode()` שנכשל → `ImageLoadError`, והמנוע מטפל בו לפי שלב 6.4.

### שלב 8 — UI (אנגלית)
- **Consent**: טקסט מ-`content/consent.js` עם `[RESEARCHER: study title]`, `[RESEARCHER: IRB/approval number]` ודומיהם, בלי לנחש תוכן. Checkbox עם הנוסח המדויק מה-PRD; `Continue` מושבת עד סימון. נרשמים `consentGiven: true` ו-`consentTimestamp`. במצב פיתוח יש כפתור "Bypass consent (dev)", שרושם `consentGiven: false, consentBypassed: true`.
- **Instructions**: פנים אחת בכל פעם, קבוצה A/B, ארבעת הכפתורים, ופידבק בניסיונות הראשונים. **בלי** אזכור של תזוזה. הטקסט ב-`content/instructions.js`.
- **Trial**: תמונה אחת ממורכזת, ומתחתיה `A | Probably A | Probably B | B` בסדר קבוע. רקע ניטרלי, בלי מונים, בלי צבע קבוצה ובלי שום רמז בסחף.
- **Feedback** (אימון בלבד): `Correct` / `Incorrect` בטקסט ניטרלי.
- **Transition**: הנוסח המדויק מ-PRD §4 וכפתור `Continue`.
- **Completion**: "Thank you…", הורדה אוטומטית, וכפתור `Download data` להורדה ידנית או ניסיון חוזר.
- **Recovery**: אם ב-localStorage יש סשן `in_progress`, מוצגים "A previous session was interrupted" ושני כפתורים: `Download its data` (CSV מסומן incomplete) ו-`Start a new session`. הסשן הישן לא נמחק.
- **Errors**: באנר באנגלית לכשלי טעינה, אחסון והורדה.

### שלב 9 — אחסון (PRD §13)
- `storageContract.js`: JSDoc ל-`startSession(meta)`, `saveTrial(trial)` ו-`completeSession(completionMeta)`. כולן async ומחזירות `{ok:true}` או זורקות `StorageError {code, message, cause}` (קודים: `QUOTA_EXCEEDED`, `UNAVAILABLE`, `SERIALIZATION`, `DOWNLOAD_BLOCKED`). בנוסף `assertImplementsContract(adapter)`.
- `index.js`: `createStorage(config)` בוחר adapter לפי `config.storage.adapter`. זה המקום היחיד שמכיר מימושים.
- `localCsvStorage.js`:
  - מפתחות: `hx:index` (רשימת sessionIds), `hx:session:<id>:meta`, ו-`hx:session:<id>:trials` (מערך, נכתב מחדש בכל ניסיון; כ-500 רשומות ≈ 300KB). יש גם עותק בזיכרון.
  - `startSession`: יוצר את הרשומות **בלי** לדרוס סשנים קודמים.
  - `saveTrial`: בדיקת כפילות לפי מפתח → זיכרון → localStorage, ועדכון ב-meta של `lastTrialNumber`, `phase` ו-`counts`. בכישלון: rollback בזיכרון ו-`StorageError`. אין הורדת CSV בשלב הזה.
  - `completeSession`: כותב meta סופי ואז מוריד `<sessionId>.csv`. אם ההורדה נחסמה, מחזיר `{ok:true, downloaded:false}` והמסך מציג כפתור ידני.
  - `exportSession(id, {incomplete})` ו-`listSessions()` משמשים את מסך ה-recovery ואת הפיתוח. הם לא חלק מהחוזה המדעי.
- README יכלול אזהרה בולטת: **CSV ו-localStorage מיועדים לפיילוט בלבד ואינם מנגנון איסוף ל-Prolific.**

### שלב 10 — CSV (`csvSerializer.js`, PRD §14)
- **עמודות לכל ניסיון (סדר קבוע):** `participantId, sessionId, trialNumber, phaseTrialNumber, phase, imageId, trueGroup, pc1, pc2, stimulusAngle, currentAngleA, currentAngleB, response, binaryResponse, correct, responseTimeMs, feedbackShown, timestamp`.
- **עמודות נוספות לשחזור:** `centerA_pc1, centerA_pc2, centerA_actualAngle, centerB_pc1, centerB_pc2, centerB_actualAngle, distanceToCenter, selectionRule` (`nearestUnused`/`nearest`), `windowAccuracy` (אימון).
- **עמודות metadata חוזרות (בכל שורה):** `experimentVersion, schemaVersion, participantIdSource, sessionStartTime, sessionEndTime, consentGiven, consentTimestamp, consentBypassed, developmentMode, sessionSeed, rngAlgorithm, pcaSha256, assetsManifestHash, trainingCompletedTrials, trainingCriterionMet, driftCompletedTrials, completionStatus, terminationReason, effectiveConfigJson, developmentOverridesJson, rngStateJson`.
- JSON מסודר דטרמיניסטית (מפתחות ממוינים). escaping לפי RFC 4180 (פסיקים, מירכאות, שורות חדשות). בוליאנים `true`/`false`. זוויות במעלות [0,360). RT במילישניות. timestamps ב-ISO 8601 UTC.
- שם קובץ: `<sessionId>.csv`, או `<sessionId>_INCOMPLETE.csv` ליצוא חלקי.

### שלב 11 — זהות ו-metadata (PRD §11)
- `sessionId = crypto.randomUUID()`.
- `participantId`: `PROLIFIC_PID` מה-URL אם קיים ולא ריק (`participantIdSource: "prolific_url"`); אחרת `anon-<random>` (`"generated_anonymous"`). ה-URL המלא לא נשמר.
- metadata כוללת את כל השדות ב-PRD §11, `completionStatus` ∈ {`in_progress`, `complete`, `incomplete_dev_early_finish`, `interrupted`} ו-`terminationReason`.

### שלב 12 — מצב פיתוח (PRD §15)
- מופעל רק אם `config.development.enabled === true`, או בהרצה דרך `npm run dev` יחד עם `?dev=1` (`import.meta.env.DEV`). ב-build של משתתפים קוד ה-panel נטען ב-dynamic import בלבד, ולכן אינו מופיע ב-UI.
- הפאנל כולל: bypass consent; override של `minTrials` ו-`accuracyWindow` (שניהם); `accuracyThreshold`, `numberOfTrials`, `degreesPerTrial` ו-seed; תצוגת debug (phase, trial#, group, זוויות נומינליות ואמיתיות, imageId, חלון דיוק); "Finish early", שרושם `completionStatus: incomplete_dev_early_finish` ו-`trainingCriterionMet` לפי הערך האמיתי, לא `true`; והורדת CSV חלקי בכל רגע שיש בו נתונים.
- כל override נכנס ל-`developmentOverrides` ול-`effectiveConfig`.

### שלב 13 — ויזואליזציה לחוקרת (PRD §Scientific visualization)
- מימוש ב-SVG (`pcaPlot.js`) שמשחזר את הקונבנציות של `create_prediction_scatter` ו-`plot_clusters_with_given_indices` (העתקה מתועדת של הפרמטרים לקובץ, לא import): כל הנקודות באפור (alpha 0.3); נקודות הטרג'קטוריה (ש-ג); מעגל מקווקו ברדיוס `max(r)·1.05`; קווים רדיאליים ותוויות כל 20°; צירים; grid; יחס 1:1; PC1/PC2.
- נקודות הטרג'קטוריה (ש-ג): איחוד התמונות שהן nearest למרכזי A ו-B בכל מיקומי הסחף לפי הקונפיג האפקטיבי, בלי החרגה (251 תמונות בברירת המחדל). התמונות שהוצגו בפועל מגיעות מהרשומות.
- שכבות: מרכזי A (× כחול) ו-B (* אדום) הנוכחיים; כל הגירויים שהוצגו (A כחול, B אדום, מלאים); הגירוי **הנוכחי** מודגש (טבעת עבה/גדול יותר); מעגל ברדיוס המרכז.
- הנתונים נלקחים **מרשומות הניסיונות בפועל** (localStorage), לא מדוגמה.
- זמינות:
  1. `review.html` — בוחר סשן מ-`hx:index` או טוען CSV שיוצא; מתעדכן חי דרך אירוע `storage` כשהניסוי רץ בטאב אחר.
  2. בתוך פאנל הפיתוח (במצב פיתוח בלבד).
- פילטרים: phase (training/drift/all) ו-group (A/B/all), ו-hover/לחיצה שמציגים imageId, זווית, trial#, תגובה ותמונה ממוזערת.
- יצוא: `Export SVG` ו-`Export PNG` (canvas, ‎2×), עם שם קובץ `<sessionId>_pca_plot.(svg|png)`. הקבצים יורדים מהדפדפן; קוד הוויזואליזציה כולו תחת `human_experiment/`.
- `review.html` לא מקושר מדף המשתתף.

### שלב 14 — בדיקות (מיפוי לבדיקות הקבלה ב-PRD §17)

| # | בדיקה | איך |
| --- | --- | --- |
| 1 | הריפו לא השתנה | `npm run verify:repo` משווה ל-baseline |
| 2 | התאמה מדעית | `make_parity_fixtures.py` (עותק של `create_base_and_opposite_points`, `rotate_vector`, `collect_nearest_images` ו-`used_indices`, על ה-CSV האמיתי) מייצר fixtures: base point ו-`−A` ל-0°, 37°, 271.5°; רצף nearest-unused של 60 ניסיונות אימון לכל קבוצה, ואחריו 360 ניסיונות סחף (סדר קבוצות קבוע מתוך fixture) בצעד 1° וב-0.5°, עם קבוצת `used` נפרדת לסחף שמתחילה ריקה. Vitest משווה (מיקום ≤1e-12, אינדקסים זהים) ובודק שוויונות קרובים |
| 2ב | החלטות החוקרת | k=1; כל `imageId` ייחודי בתוך האימון וכל `imageId` ייחודי בתוך הסחף; תמונה מהאימון **יכולה** להופיע בסחף, ורצף הסחף זהה ללא קשר לאורך האימון; תמונה שנכשלה בטעינה לא נספרת כ-used; B = −A; קונפיג עם הפרש שאינו 180° נדחה |
| 3 | פנים אחת וארבעה כפתורים | jsdom: `querySelectorAll('img').length===1`, סדר הטקסטים, שמירת `response` כפי שהוא |
| 4 | קריטריון אימון | יחידה: 19 נכונים → לא עובר; 16/20 עובר; 15/20 נכשל; 40 ניסיונות עם 30 נכונים מוקדמים וחלון אחרון 15/20 → לא עובר; מרכזים קבועים לאורך האימון |
| 5 | פידבק ואז מעבר | אינטגרציה: הניסיון המכריע מקבל פידבק → `transition` → אין ניסיון סחף עד `continue()` |
| 6 | התקדמות הסחף | אינטגרציה עם fakes: ניסיון 1 = 0°/180°, 2 = 1°/181°, 180 = 179°/359°, 181 = 180°/0°, 360 = 359°/179°; בדיוק 360 רשומות; מרכזים פנימיים חוזרים ל-0/180; זוויות עצמאיות (למשל 30°/210°) וצעדים חלופיים (100×0.5°) |
| 7 | אין פידבק בסחף | `feedbackShown=false`, `correct` מחושב, ה-DOM ללא טקסט או צבע פידבק |
| 8 | תזמון וכשלים | presenter מזויף עם עיכוב טעינה של 2s → RT לא כולל את העיכוב; כשל טעינה → אין רשומה ואין סיבוב; שתי לחיצות → רשומה אחת |
| 9 | שחזוריות | אותו seed + אותן תגובות → אותם groups ו-imageIds; איזון בסחף ≤1 ובכל בלוק אימון |
| 10 | אחסון ושחזור | localStorage מזויף: שמירה, "רענון" (מנוע חדש), recovery ויצוא; quota error → שגיאה גלויה |
| 11 | CSV | parse חוזר: שורה לכל תגובה, כל העמודות, escaping, יצוא חלקי מסומן |
| 12 | זהות ו-consent | עם ובלי `PROLIFIC_PID`; כפתור מושבת; bypass נרשם |
| 13 | בלי dev במצב משתתף; adapter ניתן להחלפה | `build` + grep ל-dist; בדיקה עם `memoryStorage` שמממש את החוזה בלי לשנות את המנוע |
| — | Playwright (smoke) | ריצה מלאה עם overrides קצרים בדפדפן אמיתי, כולל הורדת CSV |

### שלב 15 — README ומסירה (PRD §17)
README יכלול: הוראות התקנה, הכנת נכסים, הרצה ו-build; טבלת קונפיגורציה; סכמת ניסיון וסשן עם יחידות; טבלת provenance (קובץ מקור ושורות ↔ מודול מועתק); ממצאי סעיף 1 והשאלות הפתוחות מסעיף 2 עם ההחלטות שהתקבלו; נוהל ה-onset; מדיניות האיזון; תוצאות בדיקות הקבלה; ואזהרה מפורשת שהגרסה **לא מוכנה לאיסוף ב-Prolific** בלי adapter מבוסס שרת.

---

## 6. סדר ביצוע מומלץ ונקודות עצירה

1. שלבים 0–1. ההחלטות המדעיות העיקריות (ה-1 עד ה-6) כבר התקבלו; השאלות בסעיף 2.3 ממומשות עם ברירות המחדל המסומנות, וניתנות לשינוי בקונפיג בלבד.
2. שלבים 2–5 עם בדיקות יחידה ו-parity.
3. שלבים 6–9 עם בדיקות אינטגרציה.
4. שלבים 10–12.
5. שלב 13.
6. שלבים 14–15 ו-`verify:repo` סופי.
