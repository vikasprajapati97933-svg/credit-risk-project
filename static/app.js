const form = document.querySelector("#loan-form");
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const emptyResult = document.querySelector("#empty-result");
const resultContent = document.querySelector("#result-content");
const resultStatus = document.querySelector("#result-status");
const riskPill = document.querySelector("#risk-pill");
const gaugeProgress = document.querySelector("#gauge-progress");
const toast = document.querySelector("#toast");
const amountInput = form.elements.loan_amnt;
const incomeInput = form.elements.person_income;
let toastTimeout;

function updateRatioPreview() {
  const incomeValue = incomeInput.value.trim();
  const amount = Number(amountInput.value);
  if (!incomeValue || !Number.isFinite(Number(incomeValue)) || Number(incomeValue) <= 0) {
    document.querySelector("#ratio-preview").textContent = "—";
    return;
  }
  const ratio = (amount / Number(incomeValue)) * 100;
  document.querySelector("#ratio-preview").textContent = `${ratio.toFixed(1)}%`;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.classList.toggle("is-loading", isLoading);
  submitButton.querySelector(".button-label").textContent = isLoading
    ? "Assessing application"
    : "Assess application";
}

function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(toastTimeout);
  toastTimeout = window.setTimeout(() => toast.classList.remove("is-visible"), 4500);
}

function clearError() {
  formError.hidden = true;
  formError.textContent = "";
  toast.classList.remove("is-visible");
}

function showResult(result) {
  const probability = Number(result.default_probability);
  const repaymentProbability = Number(
    result.repayment_probability ?? (1 - probability),
  );
  const threshold = Number(result.threshold);
  const isHighRisk = Number(result.default_prediction) === 1;
  const defaultPercent = Math.max(0, Math.min(100, probability * 100));
  const repaymentPercent = Math.max(0, Math.min(100, repaymentProbability * 100));
  const circumference = 220;

  emptyResult.hidden = true;
  resultContent.hidden = false;
  resultStatus.textContent = isHighRisk ? "REVIEW RECOMMENDED" : "LOWER RISK";
  resultStatus.className = `result-status ${isHighRisk ? "is-high" : "is-low"}`;
  document.querySelector("#probability-value").textContent =
    `${repaymentPercent.toFixed(2)}%`;
  document.querySelector("#default-value").textContent =
    `${defaultPercent.toFixed(2)}%`;
  document.querySelector("#risk-result").textContent = result.Result;
  document.querySelector("#result-summary").textContent = isHighRisk
    ? `Default risk (${defaultPercent.toFixed(2)}%) meets or exceeds the model's fixed cutoff.`
    : `Default risk (${defaultPercent.toFixed(2)}%) is below the model's fixed cutoff.`;
  document.querySelector("#threshold-value").textContent =
    `${(threshold * 100).toFixed(2)}%`;
  document.querySelector("#threshold-marker").style.left =
    `${Math.max(0, Math.min(100, threshold * 100))}%`;
  document.querySelector("#probability-marker").style.left = `${defaultPercent}%`;
  riskPill.classList.toggle("is-high", isHighRisk);
  gaugeProgress.classList.toggle("is-high", isHighRisk);
  document.querySelector("#probability-marker").classList.toggle("is-high", isHighRisk);
  gaugeProgress.style.strokeDashoffset =
    String(circumference * (1 - repaymentPercent / 100));
}

form.addEventListener("input", updateRatioPreview);
updateRatioPreview();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  if (!form.reportValidity()) return;

  const formData = new FormData(form);
  const income = Number(formData.get("person_income"));
  const amount = Number(formData.get("loan_amnt"));
  if (!Number.isFinite(income) || income <= 0) {
    showError("Enter an annual income greater than zero.");
    incomeInput.focus();
    return;
  }
  const payload = {
    person_age: Number(formData.get("person_age")),
    person_income: income,
    person_home_ownership: formData.get("person_home_ownership"),
    person_emp_length: Number(formData.get("person_emp_length")),
    loan_intent: formData.get("loan_intent"),
    loan_grade: formData.get("loan_grade"),
    loan_amnt: amount,
    loan_int_rate: Number(formData.get("loan_int_rate")),
    loan_percent_income: amount / income,
    cb_person_default_on_file: formData.get("cb_person_default_on_file"),
    cb_person_cred_hist_length: Number(formData.get("cb_person_cred_hist_length")),
  };

  setLoading(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = "The assessment could not be completed. Check the details and try again.";
      try {
        const errorBody = await response.json();
        if (typeof errorBody.detail === "string") detail = errorBody.detail;
      } catch {
        // Keep the readable default when the server does not return JSON.
      }
      throw new Error(detail);
    }

    showResult(await response.json());
  } catch (error) {
    showError(error instanceof Error ? error.message : "Something went wrong. Please try again.");
  } finally {
    setLoading(false);
  }
});
