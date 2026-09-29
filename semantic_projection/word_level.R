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

library(lme4)
library(lmerTest)
library(clubSandwich)
library(emmeans)
library(performance)
library(ggplot2)
data <- read.csv(file.path(semantic_dir, "semantic_projection_primary.csv"), stringsAsFactors = FALSE)
data$condition <- factor(data$condition)
data$condition <- relevel(data$condition, ref = "3")

model_words <- lmer(
  projection_fear_vs_calm ~ condition + (1 | Participant),
  data = data
)

# Model fit diagnostics
diag_summary <- summary(model_words)
diag_singularity <- check_singularity(model_words)
diag_convergence <- check_convergence(model_words)
diag_hetero <- check_heteroscedasticity(model_words)
diag_normality <- check_normality(model_words)
diag_outliers <- check_outliers(model_words)
diag_r2 <- r2_nakagawa(model_words)
diag_perf <- model_performance(model_words)

print(diag_summary)
print(diag_singularity)
print(diag_convergence)
print(diag_hetero)
print(diag_normality)
print(diag_outliers)
print(diag_r2)
print(diag_perf)

# Basic residual diagnostic plots
diag_df <- data.frame(
  fitted = fitted(model_words),
  resid = resid(model_words)
)

resid_fitted_plot <- ggplot(diag_df, aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Residuals vs Fitted",
    x = "Fitted values",
    y = "Residuals"
  ) +
  theme_minimal()

qq_plot <- ggplot(diag_df, aes(sample = resid)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  labs(
    title = "Q-Q Plot of Residuals",
    x = "Theoretical quantiles",
    y = "Sample quantiles"
  ) +
  theme_minimal()

print(resid_fitted_plot)
print(qq_plot)

coef_tab <- coef_test(
  model_words,
  vcov = "CR2"
)
print(coef_tab)

emm_words <- emmeans(model_words, ~ condition)
emm_words_pairs <- pairs(emm_words)
print(emm_words)
print(emm_words_pairs)

beta_names <- names(fixef(model_words))

V <- vcovCR(
  model_words,
  cluster = model.frame(model_words)$Participant,
  type = "CR2"
)

L_pairwise <- list(
  "1 vs ref" = c(0, 1, 0),
  "2 vs ref" = c(0, 0, 1),
  "1 vs 2"   = c(0, 1, -1)
)

pw_tests <- lapply(L_pairwise, function(L) {
  Wald_test(
    model_words,
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
sig_df <- subset(sig_df, !is.na(label))

# Sensitivity analysis: lm + CR2 clustered by Participant
model_lm <- lm(
  projection_fear_vs_calm ~ condition,
  data = data
)

V_lm <- vcovCR(
  model_lm,
  cluster = data$Participant,
  type = "CR2"
)

coef_tab_lm <- coef_test(
  model_lm,
  vcov = V_lm
)
print(coef_tab_lm)

emm_lm <- emmeans(model_lm, ~ condition)
emm_lm_pairs <- pairs(emm_lm)
print(emm_lm)
print(emm_lm_pairs)

beta_names_lm <- names(coef(model_lm))
pw_tests_lm <- lapply(L_pairwise, function(L) {
  Wald_test(
    model_lm,
    constraints = matrix(L, nrow = 1, dimnames = list(NULL, beta_names_lm)),
    vcov = V_lm,
    test = "chi-sq"
  )
})

comp_df_mixed <- data.frame(
  comparison = names(pw_tests),
  p_mixed_CR2 = sapply(pw_tests, function(x) x$p_val),
  stringsAsFactors = FALSE
)
comp_df_mixed$p_mixed_CR2_bonferroni <- p.adjust(comp_df_mixed$p_mixed_CR2, method = "bonferroni")
comp_df_lm <- data.frame(
  comparison = names(pw_tests_lm),
  p_lm_CR2 = sapply(pw_tests_lm, function(x) x$p_val),
  stringsAsFactors = FALSE
)
comp_df_lm$p_lm_CR2_bonferroni <- p.adjust(comp_df_lm$p_lm_CR2, method = "bonferroni")
comparison_check <- merge(comp_df_mixed, comp_df_lm, by = "comparison", all = TRUE)
print(comparison_check)

# Standardized model-estimated condition means (residual-SD scaled)
resid_sd <- sigma(model_words)
means_summary <- as.data.frame(summary(emm_words, infer = c(TRUE, TRUE)))
means_df <- data.frame(
  condition = sub("^condition", "", means_summary$condition),
  mean = means_summary$emmean / resid_sd,
  lower = means_summary$lower.CL / resid_sd,
  upper = means_summary$upper.CL / resid_sd,
  stringsAsFactors = FALSE
)

means_df$condition <- factor(as.character(means_df$condition), levels = c("3", "1", "2"))

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
  annot_df$y <- y_max + seq_len(nrow(annot_df)) * 0.18
} else {
  annot_df <- data.frame()
}

cond_colors <- c(
  "1" = "#8de5a1",
  "2" = "#ffb482",
  "3" = "#a1c9f4"
)

p <- ggplot(means_df, aes(x = condition, y = mean, color = condition)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray45") +
  geom_point(size = 3.2) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15) +
  scale_color_manual(
    values = cond_colors,
    breaks = c("1", "2", "3"),
    labels = c("VR+Meditation", "VR Only", "Control")
  ) +
  scale_x_discrete(
    breaks = c("3", "1", "2"),
    labels = c("Control", "VR+Meditation", "VR Only")
  ) +
  labs(
    x = NULL,
    y = "Standardized projection"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 15),
    axis.text.x = element_text(size = 15),
    axis.text.y = element_text(size = 15)
  )

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
        y = y + 0.04,
        label = label
      ),
      inherit.aes = FALSE,
      size = 5,
      color = "black"
    )
}

p


ggsave(file.path(semantic_dir, "semantic_projection_primary_final.pdf"), plot = p, width = 10, height = 5)
ggsave(file.path(semantic_dir, "semantic_projection_primary_final.svg"), plot = p, width = 10, height = 5)

# -----------------------------
# 7) Reporting outputs
# -----------------------------
coef_mixed_df <- as.data.frame(coef_tab)
coef_lm_df <- as.data.frame(coef_tab_lm)

write.csv(coef_mixed_df, file.path(semantic_dir, "coefficients_mixed_cr2_primary.csv"), row.names = FALSE)
write.csv(coef_lm_df, file.path(semantic_dir, "coefficients_lm_cr2_primary.csv"), row.names = FALSE)
write.csv(comparison_check, file.path(semantic_dir, "pairwise_comparison_mixed_vs_lm_cr2_primary.csv"), row.names = FALSE)
write.csv(pairwise_raw_df, file.path(semantic_dir, "significant_pairwise_findings_primary.csv"), row.names = FALSE)
write.csv(means_df, file.path(semantic_dir, "effect_sizes_primary.csv"), row.names = FALSE)

# Text report
report_lines <- c(
  "Word-level primary Qwen relief-distress projection analysis report",
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "Saved tables:",
  "- coefficients_mixed_cr2_primary.csv",
  "- coefficients_lm_cr2_primary.csv",
  "- pairwise_comparison_mixed_vs_lm_cr2_primary.csv",
  "- significant_pairwise_findings_primary.csv",
  "- effect_sizes_primary.csv",
  "",
  "Mixed-model coefficients (CR2):",
  capture.output(print(coef_mixed_df, row.names = FALSE)),
  "",
  "Mixed-model emmeans:",
  capture.output(print(emm_words)),
  "",
  "Mixed-model emmeans pairwise comparisons:",
  capture.output(print(emm_words_pairs)),
  "",
  "Sensitivity LM coefficients (CR2):",
  capture.output(print(coef_lm_df, row.names = FALSE)),
  "",
  "Sensitivity LM emmeans:",
  capture.output(print(emm_lm)),
  "",
  "Sensitivity LM emmeans pairwise comparisons:",
  capture.output(print(emm_lm_pairs)),
  "",
  "Pairwise contrasts from CR2 Wald tests:",
  "Adjusted p values use Bonferroni correction across the 3 planned contrasts.",
  capture.output(print(pairwise_raw_df, row.names = FALSE)),
  "",
  "Mixed-model standardized condition means (residual-SD scaled):",
  capture.output(print(means_df, row.names = FALSE)),
  "",
  "Model comparison of pairwise p values (mixed model vs sensitivity LM):",
  capture.output(print(comparison_check, row.names = FALSE))
)

writeLines(report_lines, file.path(semantic_dir, "analysis_report_primary.txt"))
cat(paste(report_lines, collapse = "\n"), "\n")



# Save diagnostics report
writeLines(c(
  "Model diagnostics report",
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "summary(model_words):",
  capture.output(diag_summary),
  "",
  "check_singularity(model_words):",
  capture.output(diag_singularity),
  "",
  "check_convergence(model_words):",
  capture.output(diag_convergence),
  "",
  "check_heteroscedasticity(model_words):",
  capture.output(diag_hetero),
  "",
  "check_normality(model_words):",
  capture.output(diag_normality),
  "",
  "check_outliers(model_words):",
  capture.output(diag_outliers),
  "",
  "r2_nakagawa(model_words):",
  capture.output(diag_r2),
  "",
  "model_performance(model_words):",
  capture.output(diag_perf)
), file.path(semantic_dir, "diagnostics_report_primary.txt"))
