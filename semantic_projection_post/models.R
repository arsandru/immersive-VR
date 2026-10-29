# Condition comparison at ONE timepoint (pre or post; detected from the folder name), as in
# semantic_projection/word_level.R and mean_level.R:
#   descriptor level: projection ~ condition + (1 | participant), CR2 cluster-robust, pairwise Wald tests
#   participant level: mean projection ~ condition, pairwise contrasts (BH-FDR), Cohen's d
# Usage: Rscript models.R [subset: all|noreview]   (Portuguese-anchor axis)
suppressPackageStartupMessages({ library(lme4); library(lmerTest); library(clubSandwich); library(emmeans); library(ggplot2) })
args <- commandArgs(trailingOnly = TRUE)
subset <- if (length(args) >= 1) args[1] else "all"
script_args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", script_args[grep("--file=", script_args)])))
tm <- sub(".*_", "", basename(here))
stopifnot(tm %in% c("pre", "post"))
tag <- subset
sink(file.path(here, paste0("log_", tag, ".txt")), split = TRUE)  # full console output is also written to log_<subset>.txt

d <- read.csv(file.path(here, "projection_descriptors.csv"), stringsAsFactors = FALSE)
d$y <- d$projection
if (subset == "noreview") d <- d[!(as.character(d$needs_review) %in% c("True", "TRUE")), ]
d$participant <- factor(d$participant_id)
d$condition <- relevel(factor(d$condition_label), ref = "Control")
# Weighted effect coding (WEC) of condition: the intercept is the sample-weighted grand mean (weights = n descriptors) and each
# coefficient is that group's deviation from it; the reference group (Control) is coded -n_k / n_Control. The model is equivalent to the
# treatment-coded one (same fit, same pairwise contrasts); only the coefficients' meaning changes.
nk <- table(d$condition)
nA <- as.numeric(nk["VR Art"]); nO <- as.numeric(nk["VR Only"]); nC <- as.numeric(nk["Control"])
d$wA <- ifelse(d$condition == "VR Art", 1, ifelse(d$condition == "Control", -nA / nC, 0))
d$wO <- ifelse(d$condition == "VR Only", 1, ifelse(d$condition == "Control", -nO / nC, 0))
m <- lmer(y ~ wA + wO + (1 | participant), data = d)
V <- vcovCR(m, cluster = d$participant, type = "CR2")
mt <- lmer(y ~ condition + (1 | participant), data = d)  # treatment-coded twin: emmeans / figure and an equivalence check
stopifnot(max(abs(fitted(m) - fitted(mt))) < 1e-6, abs(sigma(m) - sigma(mt)) < 1e-6)  # same fit; REML logLik is not comparable across codings

cat("\n----- Model specification -----\n")
cat("descriptor level : y ~ condition + (1 | participant), lmer (REML), condition WEC-coded (weighted effect coding; reference = Control; weights = n descriptors per group)\n")
cat("inference        : CR2 cluster-robust (clubSandwich::vcovCR, cluster = participant); coef_test Satterthwaite; contrasts by Wald_test(test = 'HTZ'); BH-FDR over 3 pairs\n")
cat("participant level: lm(mean projection ~ condition) (treatment-coded; omnibus test and pairs equal to the WEC model); emmeans pairs, adjust = 'fdr'; Cohen's d with pooled SD\n")
cat("subset           :", subset, "\n")
print(table(condition = d$condition, useNA = "ifany"))
cat("participants per condition:\n"); print(table(unique(d[, c("participant", "condition")])$condition))
cat("\n----- lmer summary -----\n"); print(summary(m))
vc <- as.data.frame(VarCorr(m))
cat("\nICC (participant) =", round(vc$vcov[vc$grp == "participant"] / sum(vc$vcov), 3), "\n")
cat("singular fit:", isSingular(m), "\n")
cat("\n----- residual diagnostics (descriptor level) -----\n")
r <- resid(m)
cat("residual SD =", round(sd(r), 4), "\n"); print(shapiro.test(r))
re <- ranef(m)$participant[, 1]; cat("random-intercept SD =", round(sd(re), 4), "\n"); print(shapiro.test(re))
pdf(file.path(here, paste0("diagnostics_", tm, "_", tag, ".pdf")), width = 10, height = 4.5)
par(mfrow = c(1, 2)); qqnorm(r, main = "Residuals (descriptor level)"); qqline(r)
plot(fitted(m), r, xlab = "Fitted", ylab = "Residual", main = "Residuals vs fitted"); abline(h = 0, lty = 2)
dev.off()
b <- fixef(m); nm <- names(b)
cat("\n=====", tm, "(", subset, ") descriptors:", nrow(d), " participants:", nlevels(droplevels(d$participant)), "\n")
print(coef_test(m, vcov = V))
# b = (weighted grand mean, dev_VR Art, dev_VR Only); dev_Control = -(nA * dev_Art + nO * dev_Only) / nC
L <- list("VR Art vs Control" = c(0, 1 + nA / nC, nO / nC), "VR Only vs Control" = c(0, nA / nC, 1 + nO / nC), "VR Art vs VR Only" = c(0, 1, -1))
res <- do.call(rbind, lapply(names(L), function(n) {
  l <- L[[n]]; est <- sum(l * b)
  w <- Wald_test(m, constraints = matrix(l, nrow = 1, dimnames = list(NULL, nm)), vcov = V, test = "HTZ")
  se <- abs(est) / sqrt(w$Fstat); tc <- qt(0.975, w$df_denom)
  data.frame(contrast = n, estimate = est, ci_low = est - tc * se, ci_high = est + tc * se, df = w$df_denom, p_value = w$p_val)
}))
res$p_fdr <- p.adjust(res$p_value, method = "BH")
em <- as.data.frame(emmeans(mt, ~ condition))
write.csv(res, file.path(here, paste0("models_descriptor_level_", tag, ".csv")), row.names = FALSE)
write.csv(data.frame(condition = em$condition, estimated_mean = em$emmean, ci_low = em$lower.CL, ci_high = em$upper.CL),
          file.path(here, paste0("models_descriptor_means_", tag, ".csv")), row.names = FALSE)
print(format(res, digits = 3), row.names = FALSE)
cat("\nWEC: weighted grand mean =", round(b[1], 4), "; group deviations from it (CR2 Wald, 95% CI):\n")
Dv <- list("Control" = c(0, -nA / nC, -nO / nC), "VR Art" = c(0, 1, 0), "VR Only" = c(0, 0, 1))
dev <- do.call(rbind, lapply(names(Dv), function(n) {
  l <- Dv[[n]]; est <- sum(l * b); w <- Wald_test(m, constraints = matrix(l, nrow = 1, dimnames = list(NULL, nm)), vcov = V, test = "HTZ")
  se <- abs(est) / sqrt(w$Fstat); tc <- qt(0.975, w$df_denom)
  data.frame(group = n, n_descriptors = as.integer(nk[n]), deviation = est, ci_low = est - tc * se, ci_high = est + tc * se, p_value = w$p_val)
}))
print(format(dev, digits = 3), row.names = FALSE)
write.csv(dev, file.path(here, paste0("models_descriptor_level_wec_deviations_", tag, ".csv")), row.names = FALSE)

# ---- model-based figure (same design as semantic_projection/word_level.R -> semantic_projection_primary_final.pdf):
# model-estimated condition means standardised by the residual SD of the mixed model, 95% CI, dashed line at 0,
# brackets for pairs with BH-FDR adjusted p < .05 (CR2 Wald tests).
resid_sd <- sigma(m)
means_df <- data.frame(condition = factor(as.character(em$condition), levels = c("Control", "VR Art", "VR Only")),
                       mean = em$emmean / resid_sd, lower = em$lower.CL / resid_sd, upper = em$upper.CL / resid_sd)
cond_colors <- c("VR Art" = "#8de5a1", "VR Only" = "#ffb482", "Control" = "#a1c9f4")
p <- ggplot(means_df, aes(x = condition, y = mean, color = condition)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray45") +
  geom_point(size = 3.2) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15) +
  scale_color_manual(values = cond_colors) +
  labs(x = NULL, y = "Standardized projection") +
  theme_minimal() +
  theme(legend.position = "none", axis.title = element_text(size = 18), axis.text = element_text(size = 15))
sig <- res[res$p_fdr < 0.05, ]
if (nrow(sig) > 0) {
  map1 <- c("VR Art vs Control" = "Control", "VR Only vs Control" = "Control", "VR Art vs VR Only" = "VR Art")
  map2 <- c("VR Art vs Control" = "VR Art", "VR Only vs Control" = "VR Only", "VR Art vs VR Only" = "VR Only")
  lab <- ifelse(sig$p_fdr < 0.001, "***", ifelse(sig$p_fdr < 0.01, "**", "*"))
  ann <- data.frame(x1 = unname(map1[sig$contrast]), x2 = unname(map2[sig$contrast]), label = lab,
                    y = max(means_df$upper) + seq_len(nrow(sig)) * 0.18)
  lv <- levels(means_df$condition)
  p <- p + geom_segment(data = ann, aes(x = x1, xend = x2, y = y, yend = y), inherit.aes = FALSE, color = "black") +
    geom_text(data = ann, aes(x = (match(x1, lv) + match(x2, lv)) / 2, y = y + 0.04, label = label),
              inherit.aes = FALSE, size = 5, color = "black")
}
ggsave(file.path(here, paste0("semantic_projection_", tm, "_", tag, ".pdf")), plot = p, width = 10, height = 5)
ggsave(file.path(here, paste0("semantic_projection_", tm, "_", tag, ".svg")), plot = p, width = 10, height = 5)
if (tag == "all") {  # primary run: the folder's headline figure
  ggsave(file.path(here, paste0("semantic_projection_", tm, ".pdf")), plot = p, width = 10, height = 5)
  ggsave(file.path(here, paste0("semantic_projection_", tm, ".svg")), plot = p, width = 10, height = 5)
}

pm <- aggregate(y ~ participant + condition, data = d, FUN = mean); names(pm)[3] <- "mean_projection"
am <- lm(mean_projection ~ condition, data = pm)
cat("\nParticipant-level means: F test for condition\n"); print(anova(am))
cat("\n----- lm summary and diagnostics (participant level) -----\n"); print(summary(am))
print(shapiro.test(resid(am))); print(bartlett.test(mean_projection ~ condition, data = pm))
pw <- as.data.frame(pairs(emmeans(am, ~ condition), adjust = "fdr"))
sds <- tapply(pm$mean_projection, pm$condition, sd); ns <- table(pm$condition)
dcalc <- function(a, b2) {
  ma <- mean(pm$mean_projection[pm$condition == a]); mb <- mean(pm$mean_projection[pm$condition == b2])
  sp <- sqrt(((ns[a] - 1) * sds[a]^2 + (ns[b2] - 1) * sds[b2]^2) / (ns[a] + ns[b2] - 2)); unname((ma - mb) / sp)
}
pw$cohens_d <- c(dcalc("Control", "VR Art"), dcalc("Control", "VR Only"), dcalc("VR Art", "VR Only"))
write.csv(pw, file.path(here, paste0("models_participant_level_pairs_", tag, ".csv")), row.names = FALSE)
agg <- data.frame(condition = names(ns), n = as.integer(ns), mean = as.numeric(tapply(pm$mean_projection, pm$condition, mean)),
                  sd = as.numeric(sds), F_p = anova(am)[["Pr(>F)"]][1])
write.csv(agg, file.path(here, paste0("models_participant_level_means_", tag, ".csv")), row.names = FALSE)
write.csv(pm, file.path(here, paste0("participant_means_", tag, ".csv")), row.names = FALSE)
print(pw); print(agg)

cat("\n----- sessionInfo -----\n"); print(sessionInfo())
sink()
