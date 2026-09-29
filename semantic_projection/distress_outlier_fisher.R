project_dir <- normalizePath(getwd())
if (basename(project_dir) == "semantic_projection") {
  project_dir <- dirname(project_dir)
}

semantic_dir <- file.path(project_dir, "semantic_projection")

library(dplyr)

input_path <- file.path(semantic_dir, "semantic_projection_primary.csv")

analyze_file <- function(input_path) {
  if (!file.exists(input_path)) {
    stop(sprintf("Input file not found: %s", input_path))
  }

  base_name <- tools::file_path_sans_ext(basename(input_path))
  suffix <- sub("^semantic_projection_", "", base_name)

  data <- read.csv(input_path, stringsAsFactors = FALSE)

  required_cols <- c("Participant", "condition", "projection_fear_vs_calm")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns in %s: %s", input_path, paste(missing_cols, collapse = ", ")))
  }

  data$condition <- as.character(data$condition)
  data$condition <- factor(data$condition, levels = c("3", "2", "1"))

  mad0 <- median(abs(data$projection_fear_vs_calm), na.rm = TRUE)
  multiplier <- 1.00
  upper_fence <- multiplier * mad0
  lower_fence <- -multiplier * mad0

  data$is_distress_outlier <- data$projection_fear_vs_calm < lower_fence
  data$is_relief_outlier <- data$projection_fear_vs_calm > upper_fence

  count_df <- data %>%
    group_by(condition) %>%
    summarise(
      n_words = n(),
      distress_extremes = sum(is_distress_outlier, na.rm = TRUE),
      non_distress_words = n_words - distress_extremes,
      distress_pct = 100 * distress_extremes / n_words,
      relief_extremes = sum(is_relief_outlier, na.rm = TRUE),
      non_relief_words = n_words - relief_extremes,
      relief_pct = 100 * relief_extremes / n_words,
      .groups = "drop"
    ) %>%
    arrange(condition)

  distress_table <- as.matrix(count_df[, c("distress_extremes", "non_distress_words")])
  rownames(distress_table) <- as.character(count_df$condition)

  relief_table <- as.matrix(count_df[, c("relief_extremes", "non_relief_words")])
  rownames(relief_table) <- as.character(count_df$condition)

  overall_distress_test <- fisher.test(distress_table, simulate.p.value = TRUE, B = 100000)
  overall_relief_test <- fisher.test(relief_table, simulate.p.value = TRUE, B = 100000)

  pair_defs <- list(
    `1_vs_3` = c("1", "3"),
    `2_vs_3` = c("2", "3"),
    `1_vs_2` = c("1", "2")
  )

  pair_results_distress <- lapply(names(pair_defs), function(name) {
    conds <- pair_defs[[name]]
    sub <- count_df[count_df$condition %in% conds, ]
    tab <- as.matrix(sub[, c("distress_extremes", "non_distress_words")])
    rownames(tab) <- sub$condition
    ft <- fisher.test(tab)
    data.frame(
      comparison = name,
      tail = "distress",
      condition_a = conds[[1]],
      condition_b = conds[[2]],
      p_value = ft$p.value,
      odds_ratio = unname(ft$estimate),
      conf_low = unname(ft$conf.int[1]),
      conf_high = unname(ft$conf.int[2]),
      stringsAsFactors = FALSE
    )
  })

  pair_results_relief <- lapply(names(pair_defs), function(name) {
    conds <- pair_defs[[name]]
    sub <- count_df[count_df$condition %in% conds, ]
    tab <- as.matrix(sub[, c("relief_extremes", "non_relief_words")])
    rownames(tab) <- sub$condition
    ft <- fisher.test(tab)
    data.frame(
      comparison = name,
      tail = "relief",
      condition_a = conds[[1]],
      condition_b = conds[[2]],
      p_value = ft$p.value,
      odds_ratio = unname(ft$estimate),
      conf_low = unname(ft$conf.int[1]),
      conf_high = unname(ft$conf.int[2]),
      stringsAsFactors = FALSE
    )
  })

  pairwise_distress_df <- bind_rows(pair_results_distress)
  pairwise_distress_df$bonferroni_p_value <- p.adjust(pairwise_distress_df$p_value, method = "bonferroni")

  pairwise_relief_df <- bind_rows(pair_results_relief)
  pairwise_relief_df$bonferroni_p_value <- p.adjust(pairwise_relief_df$p_value, method = "bonferroni")

  distress_words_df <- data %>%
    filter(is_distress_outlier) %>%
    select(condition, Participant, word, projection_fear_vs_calm) %>%
    arrange(condition, projection_fear_vs_calm, Participant, word)

  relief_words_df <- data %>%
    filter(is_relief_outlier) %>%
    select(condition, Participant, word, projection_fear_vs_calm) %>%
    arrange(condition, desc(projection_fear_vs_calm), Participant, word)

  counts_out <- file.path(semantic_dir, sprintf("distress_outlier_counts_%s.csv", suffix))
  pairwise_distress_out <- file.path(semantic_dir, sprintf("distress_outlier_pairwise_fisher_%s.csv", suffix))
  pairwise_relief_out <- file.path(semantic_dir, sprintf("relief_extreme_pairwise_fisher_%s.csv", suffix))
  distress_words_out <- file.path(semantic_dir, sprintf("distress_outlier_words_%s.csv", suffix))
  relief_words_out <- file.path(semantic_dir, sprintf("relief_extreme_words_%s.csv", suffix))
  report_out <- file.path(semantic_dir, sprintf("distress_outlier_fisher_report_%s.txt", suffix))

  write.csv(count_df, counts_out, row.names = FALSE)
  write.csv(pairwise_distress_df, pairwise_distress_out, row.names = FALSE)
  write.csv(pairwise_relief_df, pairwise_relief_out, row.names = FALSE)
  write.csv(distress_words_df, distress_words_out, row.names = FALSE)
  write.csv(relief_words_df, relief_words_out, row.names = FALSE)

  max_distress <- count_df %>% arrange(desc(distress_pct), condition) %>% slice(1)
  max_relief <- count_df %>% arrange(desc(relief_pct), condition) %>% slice(1)
  vr_only_row <- count_df %>% filter(condition == "2")

  summary_lines <- c(
    sprintf(
      "%s has the highest rate of distress-extreme words (%d/%d, %.1f%%).",
      c("Control", "VR Only", "Flow")[match(max_distress$condition, c("3", "2", "1"))],
      max_distress$distress_extremes,
      max_distress$n_words,
      max_distress$distress_pct
    ),
    sprintf(
      "%s has the highest rate of relief-extreme words (%d/%d, %.1f%%).",
      c("Control", "VR Only", "Flow")[match(max_relief$condition, c("3", "2", "1"))],
      max_relief$relief_extremes,
      max_relief$n_words,
      max_relief$relief_pct
    ),
    sprintf(
      "VR Only shows %d/%d distress-extreme words (%.1f%%) and %d/%d relief-extreme words (%.1f%%).",
      vr_only_row$distress_extremes,
      vr_only_row$n_words,
      vr_only_row$distress_pct,
      vr_only_row$relief_extremes,
      vr_only_row$n_words,
      vr_only_row$relief_pct
    )
  )

  display_input_path <- file.path("semantic_projection", basename(input_path))

  report_lines <- c(
    sprintf("Distress-extreme Fisher report: %s", suffix),
    sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    "",
    sprintf("Input file: %s", display_input_path),
    "Zero-centered thresholding rule:",
    sprintf("MAD0 = %.6f", mad0),
    sprintf("Multiplier = %.2f", multiplier),
    sprintf("Lower threshold = %.6f", lower_fence),
    sprintf("Upper threshold = %.6f", upper_fence),
    "",
    "Summary:",
    summary_lines,
    "",
    "Condition-level extreme counts:",
    capture.output(print(as.data.frame(count_df), row.names = FALSE)),
    "",
    "Overall Fisher test on distress-extreme 3x2 table (simulated p-value):",
    sprintf("p = %.6f", overall_distress_test$p.value),
    "",
    "Pairwise Fisher tests for distress-extreme words:",
    capture.output(print(as.data.frame(pairwise_distress_df), row.names = FALSE)),
    "",
    "Overall Fisher test on relief-extreme 3x2 table (simulated p-value):",
    sprintf("p = %.6f", overall_relief_test$p.value),
    "",
    "Pairwise Fisher tests for relief-extreme words:",
    capture.output(print(as.data.frame(pairwise_relief_df), row.names = FALSE)),
    "",
    "Distress-extreme words:",
    if (nrow(distress_words_df) == 0) "None" else capture.output(print(as.data.frame(distress_words_df), row.names = FALSE)),
    "",
    "Relief-extreme words:",
    if (nrow(relief_words_df) == 0) "None" else capture.output(print(as.data.frame(relief_words_df), row.names = FALSE))
  )

  writeLines(report_lines, report_out)
  cat(paste(report_lines, collapse = "\n"), "\n")

  invisible(list(
    suffix = suffix,
    counts = count_df,
    distress_pairwise = pairwise_distress_df,
    relief_pairwise = pairwise_relief_df,
    report = report_out
  ))
}

result <- analyze_file(input_path)
