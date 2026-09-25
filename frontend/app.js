const TRAFFIC = { Low: 1, Medium: 2, High: 3 };

function apiBase() {
  const params = new URLSearchParams(window.location.search);
  if (params.get("api")) return params.get("api").replace(/\/$/, "");
  return "http://localhost:8000";
}

function payloadFromForm(form) {
  const distance = Number(form.distance.value);
  const prep = Number(form.prep.value);
  const experience = Number(form.experience.value);
  const traffic = form.traffic.value;
  const estimated = (distance / 20) * 60;
  return {
    Distance_km: distance,
    Preparation_Time_min: prep,
    Courier_Experience_yrs: experience,
    Total_Estimated_Time: prep + estimated,
    Distance_Preparation_Interaction: distance * prep,
    Distance_Traffic_Interaction: distance * TRAFFIC[traffic],
    Estimated_Delivery_Time: estimated,
    Weather: form.weather.value,
    Traffic_Level: traffic,
    Time_of_Day: form.time.value,
    Vehicle_Type: form.vehicle.value,
  };
}

function showMessage(text, isError) {
  const el = document.getElementById("message");
  el.hidden = false;
  el.textContent = text;
  el.classList.toggle("error", Boolean(isError));
}

const VIEWS = ["home", "how", "faqs"];

function showView(name) {
  const view = VIEWS.includes(name) ? name : "home";
  document.querySelectorAll("[data-page]").forEach((el) => {
    el.hidden = el.dataset.page !== view;
  });
  document.querySelectorAll(".nav a").forEach((link) => {
    link.classList.toggle("active", link.dataset.view === view);
  });
  window.scrollTo(0, 0);
}

document.querySelector(".nav").addEventListener("click", (event) => {
  const link = event.target.closest("a[data-view]");
  if (!link) return;
  event.preventDefault();
  history.pushState(null, "", `#${link.dataset.view}`);
  showView(link.dataset.view);
});

document.querySelector(".brand").addEventListener("click", (event) => {
  event.preventDefault();
  history.pushState(null, "", "#home");
  showView("home");
});

window.addEventListener("hashchange", () => {
  showView(location.hash.replace("#", ""));
});

showView(location.hash.replace("#", "") || "home");

document.getElementById("predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = document.getElementById("predict-btn");
  const body = payloadFromForm(form);
  button.disabled = true;
  showMessage("Predicting delivery time…", false);

  try {
    const response = await fetch(`${apiBase()}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Request failed (${response.status})`);
    }
    const data = await response.json();
    const minutes = Math.round(data.predicted_delivery_time_min);
    document.getElementById("arrival-value").textContent = `${minutes} min`;
    showMessage(`Your food should arrive in about ${minutes} minutes.`, false);
  } catch (error) {
    showMessage("The prediction service is unavailable. Start the API on port 8000 and try again.", true);
  } finally {
    button.disabled = false;
  }
});
