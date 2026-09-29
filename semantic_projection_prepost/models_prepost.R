# Pre vs post semantic projection: descriptor-level mixed model with CR2 cluster-robust inference
# (as semantic_projection/word_level.R, with time and time x condition), plus a participant-level
# paired analysis of the change in mean projection.
#
# Usage: Rscript models_prepost.R [subset: all|noreview]   (Portuguese-anchor axis)
suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(clubSandwich); library(emmeans)
})
args <- commandArgs(trailingOnly = TRUE)
subset <- if (length(args) >= 1) args[1] else "all"
tag <- subset
script_args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", script_args[grep("--file=", script_args)])))

d <- read.csv(file.path(here, "projection_descriptors.csv"), stringsAsFactors = FALSE)
d$y <- d$projection
if (subset == "noreview") d <- d[!(as.character(d$needs_review) %in% c("True", "TRUE")), ]
d$participant <- factor(d$participant_id)
d$condition <- relevel(factor(d$condition_label), ref = "Control")
d$time <- relevel(factor(d$time), ref = "pre")

m <- lmer(y ~ time * condition + (1 | participant), data = d)
V <- vcovCR(m, cluster = d$participant, type = "CR2")
b <- fixef(m)
nm <- names(b)
cat("Coefficients (CR2):\n"); print(coef_test(m, vcov = V))

# coefficient names: (Intercept) timepost conditionVR Art conditionVR Only timepost:conditionVR Art timepost:conditionVR Only
vec <- function(...) { v <- setNames(numeric(length(nm)), nm); w <- list(...); for (k in names(w)) v[k] <- w[[k]]; v }
tp <- "timepost"; ca <- "conditionVR Art"; co <- "conditionVR Only"
ia <- "timepost:conditionVR Art"; io <- "timepost:conditionVR Only"
contrasts <- list(
  time = list(
    "Control: post - pre"  = vec(timepost = 1),
    "VR Art: post - pre"   = vec(timepost = 1, `timepost:conditionVR Art` = 1),
    "VR Only: post - pre"  = vec(timepost = 1, `timepost:conditionVR Only` = 1)),
  pooled = list(
    "All conditions (equal weights): post - pre" = vec(timepost = 1, `timepost:conditionVR Art` = 1/3, `timepost:conditionVR Only` = 1/3)),
  interaction = list(
    "VR Art vs Control: change difference"  = vec(`timepost:conditionVR Art` = 1),
    "VR Only vs Control: change difference" = vec(`timepost:conditionVR Only` = 1),
    "VR Art vs VR Only: change difference"  = vec(`timepost:conditionVR Art` = 1, `timepost:conditionVR Only` = -1)),
  planned = list(
    "VR (Art + Only) vs Control: change difference" = vec(`timepost:conditionVR Art` = 0.5, `timepost:conditionVR Only` = 0.5)),
  baseline = list(
    "Pre: VR Art vs Control"  = vec(`conditionVR Art` = 1),
    "Pre: VR Only vs Control" = vec(`conditionVR Only` = 1),
    "Pre: VR Art vs VR Only"  = vec(`conditionVR Art` = 1, `conditionVR Only` = -1)),
  post = list(
    "Post: VR Art vs Control"  = vec(`conditionVR Art` = 1, `timepost:conditionVR Art` = 1),
    "Post: VR Only vs Control" = vec(`conditionVR Only` = 1, `timepost:conditionVR Only` = 1),
    "Post: VR Art vs VR Only"  = vec(`conditionVR Art` = 1, `timepost:conditionVR Art` = 1,
                                     `conditionVR Only` = -1, `timepost:conditionVR Only` = -1)))
rows <- list()
for (fam in names(contrasts)) {
  fam_rows <- lapply(names(contrasts[[fam]]), function(nn) {
    L <- contrasts[[fam]][[nn]]
    est <- sum(L * b)
    w <- Wald_test(m, constraints = matrix(L, nrow = 1, dimnames = list(NULL, nm)), vcov = V, test = "HTZ")
    tcrit <- qt(0.975, w$df_denom)
    se <- abs(est) / sqrt(w$Fstat)
    data.frame(family = fam, contrast = nn, estimate = est, se = se, ci_low = est - tcrit * se,
               ci_high = est + tcrit * se, df = w$df_denom, F = w$Fstat, p_value = w$p_val)
  })
  fr <- do.call(rbind, fam_rows)
  fr$p_fdr_within_family <- p.adjust(fr$p_value, method = "BH")
  rows[[fam]] <- fr
}
res <- do.call(rbind, rows)
rownames(res) <- NULL
write.csv(res, file.path(here, paste0("models_contrasts_", tag, ".csv")), row.names = FALSE)
cat("\nContrasts (descriptor-level mixed model, CR2):\n")
print(format(res, digits = 3), row.names = FALSE)

# ---- participant-level paired analysis (participants observed at both timepoints)
pm <- aggregate(y ~ participant + condition + time, data = d, FUN = mean)
wide <- reshape(pm, idvar = c("participant", "condition"), timevar = "time", direction = "wide")
wide <- wide[complete.cases(wide), ]
wide$change <- wide$y.post - wide$y.pre
p_rows <- do.call(rbind, lapply(c("VR Art", "VR Only", "Control", "ALL"), function(cn) {
  x <- if (cn == "ALL") wide$change else wide$change[wide$condition == cn]
  tt <- t.test(x)
  wt <- suppressWarnings(wilcox.test(x, exact = FALSE))
  data.frame(condition = cn, n = length(x), mean_pre = mean(if (cn == "ALL") wide$y.pre else wide$y.pre[wide$condition == cn]),
             mean_post = mean(if (cn == "ALL") wide$y.post else wide$y.post[wide$condition == cn]),
             mean_change = mean(x), ci_low = tt$conf.int[1], ci_high = tt$conf.int[2], t = unname(tt$statistic),
             p_t = tt$p.value, p_wilcoxon = wt$p.value, dz = mean(x) / sd(x))
}))
p_rows$p_fdr_conditions <- NA
p_rows$p_fdr_conditions[1:3] <- p.adjust(p_rows$p_t[1:3], "BH")
write.csv(p_rows, file.path(here, paste0("models_participant_paired_", tag, ".csv")), row.names = FALSE)
cat("\nParticipant-level paired change (post - pre in mean projection):\n"); print(format(p_rows, digits = 3), row.names = FALSE)
am <- lm(change ~ condition, data = wide)
cat("\nDoes the mean change differ by condition? (F test)\n"); print(anova(am))
write.csv(as.data.frame(anova(am)), file.path(here, paste0("models_participant_change_by_condition_", tag, ".csv")))
pp <- pairs(emmeans(am, ~ condition), adjust = "fdr")
print(pp); write.csv(as.data.frame(pp), file.path(here, paste0("models_participant_change_pairs_", tag, ".csv")), row.names = FALSE)
write.csv(wide[, c("participant", "condition", "y.pre", "y.post", "change")],
          file.path(here, paste0("participant_change_", tag, ".csv")), row.names = FALSE)
