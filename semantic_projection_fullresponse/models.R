# Condition comparison of the full-response projection (post interview; one score per participant).
# Exploratory (not pre-registered). One row per participant, so a linear model replaces the descriptor-level mixed model:
#   projection ~ condition, condition WEC-coded (weighted effect coding: intercept = sample-weighted grand mean,
#   coefficients = deviations from it; reference Control coded -n_k / n_Control; weights = n participants).
# Pairwise contrasts (emmeans, BH-FDR over 3 pairs), omnibus F-test, Cohen's d (pooled SD), diagnostics. Full output is
# also written to log.txt. Usage: Rscript models.R
suppressPackageStartupMessages({ library(emmeans); library(ggplot2) })
script_args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", script_args[grep("--file=", script_args)])))
sink(file.path(here, "log.txt"), split = TRUE)

d <- read.csv(file.path(here, "projection_participants.csv"), stringsAsFactors = FALSE)
d$condition <- relevel(factor(d$condition_label), ref = "Control")
d$y <- d$projection
nk <- table(d$condition)
nA <- as.numeric(nk["VR Art"]); nO <- as.numeric(nk["VR Only"]); nC <- as.numeric(nk["Control"])
d$wA <- ifelse(d$condition == "VR Art", 1, ifelse(d$condition == "Control", -nA / nC, 0))
d$wO <- ifelse(d$condition == "VR Only", 1, ifelse(d$condition == "Control", -nO / nC, 0))

cat("===== post, full response (3 answers concatenated), participants:", nrow(d), "\n")
cat("model: lm(projection ~ wA + wO), WEC-coded condition (reference Control, weights = n participants); treatment-coded twin for emmeans\n")
print(table(condition = d$condition)); cat("answers per participant:\n"); print(table(d$n_answers))
cat("words per text: median", median(d$n_words), " range", range(d$n_words), "\n")

m <- lm(y ~ wA + wO, data = d)
mt <- lm(y ~ condition, data = d)
stopifnot(max(abs(fitted(m) - fitted(mt))) < 1e-9)
cat("\n----- lm summary (WEC) -----\n"); print(summary(m))
cat("weighted grand mean =", round(coef(m)[1], 4), "\n")
b <- coef(m); V <- vcov(m); df <- df.residual(m); tc <- qt(0.975, df)
lc <- function(l) { est <- sum(l * b); se <- sqrt(drop(t(l) %*% V %*% l)); c(est, est - tc * se, est + tc * se, 2 * pt(-abs(est / se), df)) }
Dv <- list("Control" = c(0, -nA / nC, -nO / nC), "VR Art" = c(0, 1, 0), "VR Only" = c(0, 0, 1))
dev <- do.call(rbind, lapply(names(Dv), function(n) { r <- lc(Dv[[n]]); data.frame(group = n, n = as.integer(nk[n]), deviation = r[1], ci_low = r[2], ci_high = r[3], p_value = r[4]) }))
cat("\nGroup deviations from the weighted grand mean:\n"); print(format(dev, digits = 3), row.names = FALSE)
write.csv(dev, file.path(here, "models_wec_deviations.csv"), row.names = FALSE)

cat("\n----- omnibus -----\n"); print(anova(mt))
pw <- as.data.frame(pairs(emmeans(mt, ~ condition), adjust = "fdr"))
ci <- as.data.frame(confint(pairs(emmeans(mt, ~ condition), adjust = "none")))
sds <- tapply(d$y, d$condition, sd)
dc <- function(a, b2) { sp <- sqrt(((nk[a] - 1) * sds[a]^2 + (nk[b2] - 1) * sds[b2]^2) / (nk[a] + nk[b2] - 2)); unname((mean(d$y[d$condition == a]) - mean(d$y[d$condition == b2])) / sp) }
pw$ci_low <- ci$lower.CL; pw$ci_high <- ci$upper.CL
pw$cohens_d <- c(dc("Control", "VR Art"), dc("Control", "VR Only"), dc("VR Art", "VR Only"))
cat("\nPairwise contrasts (Control - VR Art, Control - VR Only, VR Art - VR Only; BH-FDR over 3 pairs; CI unadjusted):\n"); print(pw)
write.csv(pw, file.path(here, "models_pairs.csv"), row.names = FALSE)
agg <- data.frame(condition = names(nk), n = as.integer(nk), mean = as.numeric(tapply(d$y, d$condition, mean)), sd = as.numeric(sds), F_p = anova(mt)[["Pr(>F)"]][1])
write.csv(agg, file.path(here, "models_means.csv"), row.names = FALSE); print(agg)

cat("\n----- diagnostics -----\n"); print(shapiro.test(resid(mt))); print(bartlett.test(y ~ condition, data = d))
pdf(file.path(here, "diagnostics.pdf"), width = 10, height = 4.5); par(mfrow = c(1, 2))
qqnorm(resid(mt), main = "Residuals"); qqline(resid(mt)); plot(fitted(mt), resid(mt), xlab = "Fitted", ylab = "Residual"); abline(h = 0, lty = 2); dev.off()

em <- as.data.frame(emmeans(mt, ~ condition)); em$condition <- factor(em$condition, levels = c("Control", "VR Art", "VR Only"))
p <- ggplot(em, aes(condition, emmean)) + geom_hline(yintercept = 0, linetype = 2, colour = "grey50") +
  geom_pointrange(aes(ymin = lower.CL, ymax = upper.CL), size = 0.7) + geom_jitter(data = d, aes(condition, y), width = 0.08, alpha = 0.35, colour = "grey30") +
  labs(x = NULL, y = "Mean projection (relief positive), 95% CI", title = "Full-response projection, post interview (exploratory)") + theme_minimal(base_size = 13)
ggsave(file.path(here, "semantic_projection_fullresponse.pdf"), p, width = 6, height = 4.5)
ggsave(file.path(here, "semantic_projection_fullresponse.svg"), p, width = 6, height = 4.5)
cat("\n----- sessionInfo -----\n"); print(sessionInfo())
sink()
