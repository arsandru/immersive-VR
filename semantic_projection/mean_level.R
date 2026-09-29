script_args <- commandArgs(trailingOnly = FALSE)
script_flag <- "--file="
script_path <- sub(script_flag, "", script_args[grep(script_flag, script_args)])

if (length(script_path) > 0) {
  script_dir <- dirname(normalizePath(script_path))
  project_dir <- dirname(script_dir)
} else {
  wd <- normalizePath(getwd())
  project_dir <- if (basename(wd) == "semantic_projection") dirname(wd) else wd
}

semantic_dir <- file.path(project_dir, "semantic_projection")
data_dir <- file.path(project_dir, "data")

library(clubSandwich)
library(emmeans)
library(ggplot2)

# -----------------------------
# 1) Load data
# -----------------------------
data <- read.csv(file.path(semantic_dir, "semantic_projection_primary_mean.csv"), stringsAsFactors = FALSE)

data$condition <- factor(data$condition)
data$condition <- relevel(data$condition, ref = "3")

# -----------------------------
# 2) Fit linear model
# -----------------------------
model_lm <- lm(
  mean_projection_fear_vs_calm ~ condition,
  data = data
)

beta_names <- names(coef(model_lm))

# Cluster for CR2 corrections (use Participant if present; otherwise ID)
if ("Participant" %in% names(data)) {
  cluster_var <- data$Participant
} else if ("ID" %in% names(data)) {
  cluster_var <- data$ID
} else {
  stop("CR2 requires a clustering variable. Add a 'Participant' or 'ID' column.")
}

V <- vcovCR(
  model_lm,
  cluster = cluster_var,
  type = "CR2"
)

coef_tab <- coef_test(
  model_lm,
  vcov = V
)
print(coef_tab)

emm_lm <- emmeans(model_lm, ~ condition)
emm_lm_pairs <- pairs(emm_lm)
print(emm_lm)
print(emm_lm_pairs)

# -----------------------------
# 3) Pairwise contrasts
# -----------------------------
L_pairwise <- list(
  "1 vs ref" = c(0, 1, 0),
  "2 vs ref" = c(0, 0, 1),
  "1 vs 2"   = c(0, 1, -1)
)

pw_tests <- lapply(L_pairwise, function(L) {
  Wald_test(
    model_lm,
    constraints = matrix(L, nrow = 1, dimnames = list(NULL, beta_names)),
    vcov = V,
    test = "chi-sq"
  )
})

pairwise_raw_df <- data.frame(
  comparison = names(pw_tests),
  raw_p_value = sapply(pw_tests, function(x) x$p_val),
  stringsAsFactors = FALSE
)
pairwise_raw_df$bonferroni_p_value <- p.adjust(pairwise_raw_df$raw_p_value, method = "bonferroni")

sig_df <- pairwise_raw_df
sig_df$label <- ifelse(sig_df$bonferroni_p_value < 0.001, "***",
                ifelse(sig_df$bonferroni_p_value < 0.01,  "**",
                ifelse(sig_df$bonferroni_p_value < 0.05,  "*", NA_character_)))

# Keep only significant comparisons for plot annotation
sig_df <- subset(sig_df, !is.na(label))

# -----------------------------
# 4) Build means_df
# -----------------------------
means_df <- aggregate(
  mean_projection_fear_vs_calm ~ condition,
  data = data,
  FUN = function(x) c(
    mean = mean(x, na.rm = TRUE),
    SE = sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x))),
    n = sum(!is.na(x))
  )
)

means_df <- do.call(data.frame, means_df)
names(means_df) <- c("condition", "mean", "SE", "n")
means_df$t_crit <- qt(0.975, df = means_df$n - 1)
means_df$lower <- means_df$mean - means_df$t_crit * means_df$SE
means_df$upper <- means_df$mean + means_df$t_crit * means_df$SE
means_df$condition <- factor(as.character(means_df$condition), levels = c("3", "1", "2"))

ci_summary <- means_df[, c("condition", "n", "mean", "SE", "t_crit", "lower", "upper")]
print(ci_summary)

# -----------------------------
# 5) Build annotation data
# -----------------------------
if (nrow(sig_df) > 0) {
  map_x1 <- c("1 vs ref" = "3", "2 vs ref" = "3", "1 vs 2" = "1")
  map_x2 <- c("1 vs ref" = "1", "2 vs ref" = "2", "1 vs 2" = "2")

  annot_df <- data.frame(
    comparison = sig_df$comparison,
    label = sig_df$label,
    x1 = unname(map_x1[sig_df$comparison]),
    x2 = unname(map_x2[sig_df$comparison]),
    stringsAsFactors = FALSE
  )

  y_max <- max(means_df$upper, na.rm = TRUE)
  annot_df$y <- y_max + seq_len(nrow(annot_df)) * 2
} else {
  annot_df <- data.frame()
}

# -----------------------------
# 6) Plot
# -----------------------------
cond_colors <- c(
  "1" = "#8de5a1",
  "2" = "#ffb482",
  "3" = "#a1c9f4"
)

p <- ggplot(means_df, aes(x = condition, y = mean, color = condition)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15) +
  scale_color_manual(
    values = cond_colors,
    breaks = c("1", "2", "3"),
    labels = c("Flow (VR+Meditation)", "VR Only", "Control (Care as Usual)")
  ) +
  scale_x_discrete(
    breaks = c("3", "1", "2"),
    labels = c("Control", "Flow", "VR Only")
  ) +
  labs(
    x = "Condition",
    y = "Mean relief-distress projection",
    title = "Primary Qwen condition means (CR2-adjusted)",
    subtitle = "Only Bonferroni-significant pairwise differences shown; error bars are t-based 95% CI",
    color = "Group"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

if (nrow(annot_df) > 0) {
  p <- p +
    geom_segment(
      data = annot_df,
      aes(x = x1, xend = x2, y = y, yend = y),
      inherit.aes = FALSE,
      color = "black"
    ) +
    geom_text(
      data = annot_df,
      aes(
        x = (as.numeric(factor(x1, levels = levels(means_df$condition))) +
             as.numeric(factor(x2, levels = levels(means_df$condition)))) / 2,
        y = y + 0.5,
        label = label
      ),
      inherit.aes = FALSE,
      size = 5,
      color = "black"
    )
}

print(p)


# Save plot
ggsave(file.path(semantic_dir, "semantic_projection_primary_mean.pdf"), plot = p, width = 10, height = 8)


# -----------------------------
# 7) Reporting outputs
# -----------------------------
coef_lm_df <- as.data.frame(coef_tab)
write.csv(coef_lm_df, file.path(semantic_dir, "coefficients_lm_cr2_primary_mean.csv"), row.names = FALSE)
write.csv(pairwise_raw_df, file.path(semantic_dir, "significant_pairwise_findings_primary_mean.csv"), row.names = FALSE)
write.csv(ci_summary, file.path(semantic_dir, "ci_summary_primary_mean.csv"), row.names = FALSE)

report_lines <- c(
  "Mean-level primary Qwen relief-distress projection analysis report",
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "Saved tables:",
  "- coefficients_lm_cr2_primary_mean.csv",
  "- significant_pairwise_findings_primary_mean.csv",
  "",
  "Model coefficients (CR2):",
  capture.output(print(coef_lm_df, row.names = FALSE)),
  "",
  "Model emmeans:",
  capture.output(print(emm_lm)),
  "",
  "Model emmeans pairwise comparisons:",
  capture.output(print(emm_lm_pairs)),
  "",
  "Pairwise contrasts from CR2 Wald tests:",
  "Adjusted p values use Bonferroni correction across the 3 planned contrasts.",
  capture.output(print(pairwise_raw_df, row.names = FALSE)),
  "",
  sprintf("Bonferroni-adjusted alpha for 3 contrasts: %.5f", 0.05 / 3)
)

writeLines(report_lines, file.path(semantic_dir, "analysis_report_primary_mean.txt"))
cat(paste(report_lines, collapse = "\n"), "\n")

# Diagnostics report (LM)
writeLines(c(
  "Model diagnostics report (mean-level LM)",
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "summary(model_lm):",
  capture.output(summary(model_lm))
), file.path(semantic_dir, "diagnostics_report_primary_mean.txt"))
