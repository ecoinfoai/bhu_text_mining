# Getting Started with FormA: A Guide for New Teachers

FormA (formative-analysis) is a command-line toolkit designed for university instructors who run regular formative assessments — short in-class tests, quizzes, or concept checks. It handles everything from generating exam papers and scanning answer sheets, to evaluating student responses with an LLM-assisted rubric, producing individual feedback PDFs, tracking improvement over time, and even emailing reports directly to students.

This guide is written for teachers who are new to FormA. You do not need to be a programmer to follow it. Each scenario in this guide reflects a real situation you might find yourself in — the first week of class, the middle of the semester when you suspect some students are falling behind, or the final weeks when you need grade predictions. Read through the scenario that matches where you are right now, follow the commands, and come back to the other scenarios as your semester progresses.

One note on commands: FormA uses the single `forma` entry point with subcommands, for example `forma eval` or `forma report student`. Older tutorials may show hyphenated names like `forma-eval`; those are no longer used.

---

## Scenario A: I just want to evaluate my class this week

**Who this is for:** You teach a single class section, it is your first time using FormA, and you have no prior history or configuration files. You want to run a formative test, collect the results, and hand students a personalized feedback report by the end of the week.

### The story

It is Monday morning and you have decided to give your students a short concept-check quiz at the end of today's lecture. You have a YAML file with your question bank — or you are about to create one — and you want FormA to do the heavy lifting: print exam sheets, scan the handwritten responses, evaluate them using AI, and produce a PDF for each student that explains what they got right, what they missed, and what to review before the next class.

This scenario walks you through the entire pipeline from zero. By the end, every student will have a personalized PDF in their inbox.

### Step 1: Initialize your project

Before anything else, run `forma init` in your course directory. This creates a `forma.yaml` configuration file that stores your course name, SMTP settings, and default file paths so you do not have to type them on every command.

```bash
forma init
```

The command will ask a few questions interactively: your course name, the semester label, your SMTP server details, and where you want to save reports. Answer them and a `forma.yaml` file appears in the current directory. You can edit it at any time with a text editor.

### Step 2: Generate exam papers

Your question bank lives in a YAML file. Pass it to `forma exam` to produce a printable exam PDF. The `--questions` flag selects which question numbers to include, and `--num-papers` controls how many copies to print.

```bash
forma exam \
  --config exams/Ch01_FormativeTest.yaml \
  --questions 1 3 \
  --num-papers 30 \
  --output week_01_exam.pdf
```

After this you will have `week_01_exam.pdf` ready to print. Hand it out, run the test, collect the papers.

### Step 3: Scan the answer sheets

Once you have collected the physical answer sheets, photograph or scan them — good flat lighting, pages lying completely flat, consistent orientation. Put all the images into a directory called (for example) `scans_1A_w1/`. Then run:

```bash
forma ocr scan --class A \
  --image-dir scans_1A_w1 \
  --output scans_1A_w1/ocr_results.yaml \
  --num-questions 2
```

FormA will extract the handwritten answers from the images and write them to `ocr_results.yaml`. If you also collected responses via Google Forms (for the online cohort, for example), you will join them in the next step.

### Step 4: Join OCR results with Google Forms data (if applicable)

If some students submitted via Google Forms, merge both sources into one clean dataset:

```bash
forma ocr join --class A \
  --ocr scans_1A_w1/ocr_results.yaml \
  --forms forms_responses_w1.csv \
  --output scans_1A_w1/final.yaml
```

If you only have scanned paper responses and no Google Forms data, you can skip the `--forms` flag and the command will still produce the `final.yaml` file from the OCR output alone.

### Step 5: Evaluate student responses

This is where the LLM evaluation happens. FormA reads your exam configuration (which contains the correct answers, concept tags, and rubric criteria) and evaluates each student response, producing a score and a written feedback comment for every question.

```bash
forma eval --class A \
  --config exams/Ch01_FormativeTest.yaml \
  --responses scans_1A_w1/final.yaml \
  --output scans_1A_w1/eval/
```

Evaluation can take a few minutes depending on how many students and questions you have. The output directory (`scans_1A_w1/eval/`) will contain one YAML file per student with their scores, concept coverage, and feedback text. There is also an aggregate `final.yaml` with class-level statistics.

If you are testing the pipeline for the first time and do not want to burn through your Gemini API quota, add `--n-calls 1` to run each question through the LLM only once rather than the default three-call reliability protocol:

```bash
forma eval --class A \
  --config exams/Ch01_FormativeTest.yaml \
  --responses scans_1A_w1/final.yaml \
  --output scans_1A_w1/eval/ \
  --n-calls 1
```

### Step 6: Generate individual student reports

Each student gets a PDF that shows their scores per concept, written feedback for every question they attempted, and a "what to review" section highlighting the concepts where they scored below the threshold.

```bash
forma report student --class A \
  --config exams/Ch01_FormativeTest.yaml \
  --eval-dir scans_1A_w1/eval/ \
  --output-dir reports/week_01/students/
```

After this command finishes you will find one PDF per student in `reports/week_01/students/`. The filename includes the student ID for easy reference.

### Step 7: Generate the professor summary report

The professor report gives you a class-level view: score distributions, the concepts most commonly missed, a ranked list of students by performance, and — if you have been running FormA long enough — longitudinal trend charts.

```bash
forma report professor --class A \
  --config exams/Ch01_FormativeTest.yaml \
  --eval-dir scans_1A_w1/eval/ \
  --output reports/week_01/professor_A.pdf
```

Open `professor_A.pdf` and you will see histograms of the score distribution, a bar chart showing which concepts had the lowest average scores, a table of all students sorted by total score, and written commentary generated from the data.

### Step 8: Prepare the email delivery package

Before you can send anything, FormA needs to bundle the reports and match each PDF to the correct student email address. Your student roster (a YAML or CSV file mapping student IDs to email addresses) must be available.

```bash
forma deliver prepare --class A \
  --report-dir reports/week_01/students/ \
  --roster roster.yaml \
  --output delivery/week_01_A/
```

This creates a delivery manifest in `delivery/week_01_A/` listing every student, their email address, and the path to their PDF. No emails are sent yet.

### Step 9: Send the emails

Always do a dry run first to make sure every address looks right and the attachments are matched correctly:

```bash
forma deliver send \
  --manifest delivery/week_01_A/manifest.yaml \
  --dry-run
```

The dry run prints what would be sent without actually connecting to your mail server. Once you are satisfied, remove `--dry-run` and set your SMTP password via the environment variable (never type it directly in the shell — see Tips and Gotchas at the end of this guide):

```bash
export FORMA_SMTP_PASSWORD="your_password_here"

forma deliver send \
  --manifest delivery/week_01_A/manifest.yaml
```

Students will receive an email with their personalized PDF attached. You are done.

---

## Scenario B: I teach multiple sections and want a unified view

**When to use this:** You have two or more class sections — call them A and B — that took the same formative test this week. You want individual reports for each section's students, but you also want a single summary that compares the two sections side by side. Maybe you want to know whether section B understood the material better than section A, or whether the difficulty felt different across the two groups.

FormA's batch pipeline handles this cleanly. You scan and evaluate each section independently, then merge them into aggregate reports with built-in statistical comparison (Welch's t-test and Mann-Whitney U, with Bonferroni correction across multiple pairs).

### Running the pipeline for each section

Scan and join each section as you normally would:

```bash
forma ocr scan --class A \
  --image-dir scans_1A_w1 \
  --output scans_1A_w1/ocr_results.yaml \
  --num-questions 3

forma ocr scan --class B \
  --image-dir scans_1B_w1 \
  --output scans_1B_w1/ocr_results.yaml \
  --num-questions 3
```

### Batch evaluation across sections

Instead of running `forma eval` twice, use `forma eval batch` to evaluate all sections in one pass. FormA will resolve the response file for each class using the pattern you provide:

```bash
forma eval batch \
  --config exams/Ch01_FormativeTest.yaml \
  --classes A B \
  --responses-pattern "scans_1{class}_w1/final.yaml" \
  --output-pattern "scans_1{class}_w1/eval/"
```

The `{class}` placeholder is replaced with each class label in turn. After this step, both `scans_1A_w1/eval/` and `scans_1B_w1/eval/` are populated.

### Batch report generation

Generate individual student reports for all sections at once:

```bash
forma report batch \
  --config exams/Ch01_FormativeTest.yaml \
  --classes A B \
  --eval-pattern "scans_1{class}_w1/eval/" \
  --output-dir reports/week_01/
```

### Aggregate professor summary

The `--aggregate` flag tells FormA to produce an additional merged professor report that covers all sections together, including the cross-section statistical comparison table:

```bash
forma report professor --aggregate \
  --config exams/Ch01_FormativeTest.yaml \
  --classes A B \
  --eval-pattern "scans_1{class}_w1/eval/" \
  --output reports/week_01/professor_aggregate.pdf
```

After this, you will have individual section PDFs (`professor_A.pdf`, `professor_B.pdf`) as well as the aggregate PDF showing which section performed better on each concept, whether the difference is statistically significant, and the effect size. This is useful for calibrating instruction if one section is consistently behind the other. See `docs/cli-reference.md` for the full list of flags including `--no-individual` if you only want the aggregate and not the per-section files.

---

## Scenario C: I want to track how students improve over the semester

**When to use this:** You have been running formative tests for several weeks and now you want to see the picture over time. Which students are steadily improving? Which ones started strong but have been slipping? Is the class as a whole trending up or down on the key concepts? This is the longitudinal view, and it requires a `longitudinal.yaml` store that accumulates results week after week.

FormA updates this store automatically whenever you run `forma eval` with the `--longitudinal-store` flag pointing at the same file. After a few weeks of data, you can train a risk prediction model and generate longitudinal trend reports.

### Building up the store week over week

Each time you run evaluation, point it at your longitudinal store:

```bash
forma eval --class A \
  --config exams/Ch02_FormativeTest.yaml \
  --responses scans_1A_w2/final.yaml \
  --output scans_1A_w2/eval/ \
  --longitudinal-store longitudinal.yaml
```

FormA appends this week's results to `longitudinal.yaml`. After four or more weeks, the store contains enough data to be interesting.

### Training the risk prediction model

The risk model uses logistic regression on a feature set extracted from the longitudinal store — things like average score trend, concept coverage decay, and missed-test flags. Train it like this:

```bash
forma train risk \
  --store longitudinal.yaml \
  --output models/risk.pkl \
  --weeks 1 2 3 4
```

The `--weeks` flag tells FormA which week columns to use as training features. The trained model is saved to `models/risk.pkl` and will be used in Scenario D.

### Generating the longitudinal report

```bash
forma report longitudinal \
  --store longitudinal.yaml \
  --config exams/Ch01_FormativeTest.yaml \
  --output reports/longitudinal_week_04.pdf
```

After this you will have a multi-page PDF showing individual student trend lines (are their weekly scores going up, flat, or down?), concept-level heatmaps across the weeks, OLS regression slopes per student, and a class-level cohort trajectory chart. This report is particularly useful to share with your department chair or to prepare mid-semester advising conversations. For full flag documentation see `docs/cli-reference.md`.

---

## Scenario D: I want to identify at-risk students early

**When to use this:** It is week four or five of the semester. A few students are worrying you — they seem disengaged, their scores have been erratic, or they missed a test. You want FormA to surface a ranked list of students most at risk of dropping or failing before the situation becomes irreversible. The earlier you catch this, the more time you have to intervene.

This scenario assumes you have already trained a risk model (see Scenario C). If you have not, you can still run the warning report using only the rule-based risk flags (no model required), but adding a trained model improves accuracy significantly.

### The story

Professor Kim teaches Introductory Physics. After week four she pulls up the longitudinal report and notices that three students have flat or declining trend lines. She runs the early warning report and finds that two of those students are flagged as high risk by the model, while a third is flagged by the rule-based system for having missed two consecutive weeks. Armed with this list, she emails each student personally and schedules office hours.

### Running the early warning report

```bash
forma report warning \
  --config exams/Ch01_FormativeTest.yaml \
  --eval-dir scans_1A_w4/eval/ \
  --longitudinal-store longitudinal.yaml \
  --week 4 \
  --model models/risk.pkl \
  --output reports/week_04_warning.pdf
```

The output PDF contains a dashboard showing the class risk distribution (how many students fall into each risk category), followed by individual "warning cards" for every student the system has flagged. Each card shows the student's risk score, which risk factors triggered the flag (low concept coverage, declining trend, missed test, low absolute score), and a suggested intervention type based on the risk profile.

If you do not yet have a trained model, omit the `--model` flag and FormA will use only rule-based detection. This is less precise but still useful in the first few weeks before you have enough longitudinal data to train. See `docs/cli-reference.md` for threshold tuning options.

---

## Scenario E: I started interventions — how do I track their effect?

**When to use this:** You have identified at-risk students and you have started doing something about it — scheduling one-on-one meetings, assigning supplementary problems, connecting students with peer tutors. Now you want to record what you did and measure whether it actually helped. FormA's intervention log lets you do exactly that.

### The story

After running the week four warning report, Professor Kim schedules check-in meetings with three students. She wants to record these interventions in FormA so that she can see, by week six, whether the students who received interventions improved more than those who did not. This data will also be visible in future professor reports.

### Adding an intervention record

Each intervention is recorded with a student ID, the week it happened, the type of intervention, and an optional description:

```bash
forma intervention add \
  --store intervention_log.yaml \
  --student S2024001 \
  --week 4 \
  --type 면담 \
  --description "30-minute office hours session, reviewed thermodynamics concepts"
```

The supported intervention types are: `면담` (meeting/consultation), `보충학습` (supplementary study), `과제부여` (assignment), `멘토링` (mentoring), `기타` (other).

### Listing recorded interventions

```bash
forma intervention list \
  --store intervention_log.yaml \
  --student S2024001
```

Or list all interventions for a given week across all students:

```bash
forma intervention list \
  --store intervention_log.yaml \
  --week 4
```

### Recording the outcome

After a couple of weeks have passed, come back and record what happened:

```bash
forma intervention update \
  --store intervention_log.yaml \
  --id 1 \
  --outcome 개선
```

The outcome options are `개선` (improved), `유지` (maintained), and `악화` (worsened). FormA uses a two-week window by default when computing the effect: it compares the student's performance in the two weeks after the intervention against the two weeks before.

### Seeing intervention effects in the professor report

Once you have recorded some interventions with outcomes, regenerate the professor report with the intervention log attached:

```bash
forma report professor --class A \
  --config exams/Ch01_FormativeTest.yaml \
  --eval-dir scans_1A_w6/eval/ \
  --intervention-log intervention_log.yaml \
  --output reports/week_06/professor_A.pdf
```

The report now includes an intervention effectiveness section showing which intervention types correlated with improvement, a per-student summary of interventions and their outcomes, and a bar chart comparing the average score change for intervened versus non-intervened students. This evidence is valuable for your own professional reflection and for departmental reporting. See `docs/cli-reference.md` for all available flags.

---

## Scenario F: End of semester — can FormA predict grades?

**When to use this:** You are approaching the end of the semester. You have been running formative tests and recording scores in the longitudinal store. Now you want FormA to predict each student's likely final semester grade based on their formative performance trajectory. This is useful for proactive advising: students whose predicted grade is lower than their self-assessment can be counseled before the final exam.

### The story

It is week twelve. Professor Lee has ten weeks of formative data for her Introduction to Programming class. She wants to predict which students are on track for an A, which are in the B/C range, and which are at risk of failing the course. She has a `grade_mapping.yaml` file from last year that maps historical formative scores to final grades. Using this, FormA can train a grade prediction model and produce predictions for the current cohort.

### Creating the grade mapping file

Your `grade_mapping.yaml` maps historical student records to their actual final grades. This is data from a previous semester:

```yaml
students:
  - id: "2023-S001"
    final_grade: "A"
  - id: "2023-S002"
    final_grade: "B+"
  - id: "2023-S003"
    final_grade: "C"
```

The longitudinal store for that semester must also be available. FormA extracts 21 features per student (weekly score trends, concept coverage patterns, improvement velocity, attendance proxies) and trains a logistic regression classifier.

### Training the grade prediction model

```bash
forma train grade \
  --store longitudinal_2023.yaml \
  --grades grade_mapping.yaml \
  --output models/grade.pkl \
  --semester "2023-Fall" \
  --min-students 10
```

The `--min-students` flag guards against training on too little data — if your historical cohort is smaller than this threshold, the command will warn you and refuse to train to avoid an unreliable model.

### Generating grade predictions for the current semester

With the model trained, generate an updated professor report that includes grade predictions:

```bash
forma report professor --class A \
  --config exams/Ch01_FormativeTest.yaml \
  --eval-dir scans_1A_w12/eval/ \
  --longitudinal-store longitudinal.yaml \
  --grade-model models/grade.pkl \
  --output reports/week_12/professor_A_with_grades.pdf
```

After this, you will have a professor report that includes a grade prediction table — every student's predicted grade distribution (probability of each grade band), a confidence score, and a flag for "cold start" students who have too little history for a reliable prediction.

This output is not meant to replace the actual final grade — FormA is clear that these are probabilistic estimates. But they are a powerful tool for early advising conversations and for identifying students who need one final push. For full flag documentation and confidence threshold settings, see `docs/cli-reference.md`.

---

## Scenario G: I want to automate my weekly routine with week.yaml

**When to use this:** You have been running FormA for a few weeks and you find yourself typing the same long commands with the same paths every single time. You teach a consistent weekly rhythm — same exam structure, same directory layout — and you want to encode that structure once so that FormA resolves all paths automatically. The `week.yaml` file is FormA's answer to this.

### The story

Professor Choi teaches General Chemistry and runs a formative quiz every Friday. By week three, she is tired of typing out the same twelve flags on every `forma ocr` and `forma eval` command. Her teaching assistant suggests using `week.yaml` to lock in the paths once and let FormA figure out the rest. Now her weekly routine is three short commands, and she only needs to update one number (the week number) at the top of the file each Friday.

### A complete week.yaml example

Place this file in your course directory and update the `week:` field before each session:

```yaml
week: 1

select:
  source: "../exams/Ch01_FormativeTest.yaml"
  questions: [1, 3]
  num_papers: 50
  exam_output: "week_01_exam.pdf"

ocr:
  num_questions: 2
  image_dir_pattern: "scans_1{class}_w1"
  ocr_output_pattern: "scans_1{class}_w1/ocr_results.yaml"
  join_output_pattern: "scans_1{class}_w1/final.yaml"
  join_forms_csv: "forms_responses_w1.csv"
  crop_coords: []

eval:
  config: "../exams/Ch01_FormativeTest.yaml"
  questions_used: [1, 3]
  responses_pattern: "scans_1{class}_w1/final.yaml"
  output_pattern: "scans_1{class}_w1/eval/"
  generate_reports: true
```

The `{class}` placeholder in any pattern field is replaced with the actual class label (A, B, C, ...) when you run the command with `--class`.

### Selecting questions and generating the exam

With `week.yaml` in place, run `forma select` to produce this week's exam PDF. FormA reads the `select` block and resolves all paths automatically:

```bash
forma select --week-config week.yaml
```

This reads `source`, picks `questions: [1, 3]`, prints `num_papers: 50` copies, and saves the result to `week_01_exam.pdf` — all without any additional flags.

### Scanning and joining with auto-resolved paths

The OCR commands now only need `--class` to resolve the `{class}` placeholders:

```bash
forma ocr scan --class A --week-config week.yaml
```

FormA reads `image_dir_pattern: "scans_1{class}_w1"` from `week.yaml`, substitutes `A` for `{class}`, and scans `scans_1A_w1/`. The output goes to `scans_1A_w1/ocr_results.yaml` as specified in `ocr_output_pattern`.

```bash
forma ocr join --class A --week-config week.yaml
```

This merges the OCR output with `forms_responses_w1.csv` and writes the result to `scans_1A_w1/final.yaml`.

### Evaluation with auto-resolved paths

```bash
forma eval --class A --week-config week.yaml
```

FormA reads the `eval` block: it knows which config file to use, which questions were included, where the responses live, and where to write the output. No long flag lists required.

### The payoff

Once `week.yaml` is set up, your entire weekly routine for section A becomes:

```bash
forma select --week-config week.yaml
forma ocr scan --class A --week-config week.yaml
forma ocr join --class A --week-config week.yaml
forma eval --class A --week-config week.yaml
forma report student --class A --week-config week.yaml
forma report professor --class A --week-config week.yaml
forma deliver prepare --class A --week-config week.yaml
forma deliver send --manifest delivery/week_01_A/manifest.yaml
```

And at the start of week two, you only need to change `week: 1` to `week: 2` at the top of `week.yaml` (and update path patterns if your naming convention includes the week number). The `crop_coords` field is written back to `week.yaml` automatically the first time you run a successful OCR scan, so you do not have to calibrate the scan region every week.

For the complete list of fields supported in `week.yaml`, the four-layer config merge order (project → semester → week → CLI flags), and how to use `{class}` patterns in nested paths, see `docs/cli-reference.md` and `docs/configuration.md`.

---

## Tips and Gotchas

This section collects the most common stumbling blocks new users run into, along with quick guidance on each one.

### Never type your SMTP password in the shell

If you run `forma deliver send` with your email password as a command-line argument, it will be saved in your shell history in plain text — a serious security risk. FormA never stores the password in any file. Instead, use the environment variable:

```bash
export FORMA_SMTP_PASSWORD="your_actual_password"
forma deliver send --manifest delivery/week_01_A/manifest.yaml
```

If you are in a scripted or CI environment where environment variables are not practical, you can pipe the password from stdin:

```bash
echo "your_actual_password" | forma deliver send \
  --manifest delivery/week_01_A/manifest.yaml \
  --password-from-stdin
```

Either way, the password never appears in any log file, YAML file, or shell history entry.

### Always do a dry run before sending emails

This cannot be said strongly enough: run `--dry-run` every time before you actually send. The dry run prints exactly what would be sent — the recipient address, the subject line, the attachment filename — without connecting to your mail server. It takes two seconds and it prevents you from sending 80 emails to the wrong addresses or with the wrong attachments.

```bash
forma deliver send --manifest delivery/week_01_A/manifest.yaml --dry-run
```

Only when you have read through the dry-run output and everything looks correct should you remove `--dry-run` and send for real.

### The Gemini free tier is 15 requests per minute

If you are using the default Gemini LLM provider and you have a large class, the evaluation step may hit the 15 RPM (requests per minute) rate limit and slow down significantly. During initial setup and testing, add `--n-calls 1` to `forma eval` to send only a single LLM call per question instead of the default three-call reliability protocol:

```bash
forma eval --class A \
  --config exams/Ch01_FormativeTest.yaml \
  --responses scans_1A_w1/final.yaml \
  --output scans_1A_w1/eval/ \
  --n-calls 1
```

Using `--n-calls 1` reduces evaluation quality slightly (there is no median aggregation across multiple calls) but it is perfectly adequate for a quick sanity check. For production runs where you want accurate, reliable scores, use the default (3 calls) and be patient, or upgrade to a paid Gemini tier.

### OCR scan quality requirements

The OCR step is the most sensitive to physical conditions. Poor scans are the single most common reason new users get wrong scores. Follow these guidelines:

- **Flat pages**: Paper that is even slightly curved (from being rolled or folded) will cause the perspective correction to fail. Use a flat document scanner if possible; if you are photographing with a phone, press the paper firmly onto a flat surface.
- **Good lighting**: Avoid harsh shadows across the page. Indirect, even lighting works best. Do not photograph near a window with direct sunlight.
- **Consistent orientation**: All pages should be in the same orientation (all portrait, all landscape). Do not mix orientations in the same scan directory.
- **Resolution**: Aim for at least 200 DPI. Most phone cameras at normal distance produce more than enough resolution; the problem is usually lighting and flatness, not resolution.

If OCR results look wrong (wrong student IDs extracted, missing answers), try rescanning with better lighting before assuming a bug.

### Use `--provider anthropic` for higher quality feedback

FormA supports two LLM providers: Gemini (default) and Anthropic (Claude). For most routine evaluation, Gemini is sufficient and faster. However, if you are working on a high-stakes assessment or you want richer, more nuanced written feedback for students, the Anthropic provider tends to produce more detailed and pedagogically useful commentary:

```bash
forma eval --class A \
  --config exams/Ch01_FormativeTest.yaml \
  --responses scans_1A_w1/final.yaml \
  --output scans_1A_w1/eval/ \
  --provider anthropic
```

Note that using the Anthropic provider requires an `ANTHROPIC_API_KEY` environment variable to be set. The Gemini default requires `GEMINI_API_KEY`.

### Use `--skip-stats` for a quick feedback check

The full evaluation pipeline includes a statistical analysis layer (ICC computation, concept-level reliability metrics). This layer requires at least two students and adds noticeable computation time for large classes. If you just want to quickly check that the feedback text looks reasonable on a small test run, skip it:

```bash
forma eval --class A \
  --config exams/Ch01_FormativeTest.yaml \
  --responses scans_1A_w1/final.yaml \
  --output scans_1A_w1/eval/ \
  --skip-stats \
  --n-calls 1
```

This combination (`--skip-stats --n-calls 1`) is the fastest possible evaluation run — useful when you are iterating on your exam YAML or debugging a path configuration problem.

### The full flag reference lives in docs/cli-reference.md

This guide covers the most common use cases and the flags you will need 90% of the time. Every command has many more options — threshold tuning, output format control, verbosity flags, model hyperparameters, and more. The authoritative reference is `docs/cli-reference.md`. When a command does not behave the way you expect, check there first: the answer is usually a flag you did not know existed.
