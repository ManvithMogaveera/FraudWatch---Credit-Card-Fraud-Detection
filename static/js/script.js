document.getElementById("fraudForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  // 1. Get form values
  const amount = document.getElementById("amount").value;
  const time = document.getElementById("time").value;
  const features = document.getElementById("features").value;

  const resultBox = document.getElementById("result");

  // 2. Reset the result box and show a loading state
  // This uses the "show" class to trigger the fade-in animation
  resultBox.className = "result-box show";
  resultBox.innerHTML = `
    <i class="fa-solid fa-spinner fa-spin"></i>
    Analyzing Transaction...
  `;

  // 3. Prepare data and send to the backend
  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        // Ensure data is sent as numbers where appropriate
        time: Number(time),
        amount: Number(amount),
        features: features.split(",").map(Number),
      }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }

    const data = await response.json();

    // 4. Display the result using CSS classes for styling
    if (data.prediction === 1) {
      // Add 'fraud' class to apply red styling from the CSS
      resultBox.className = "result-box show fraud";
      resultBox.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Fraudulent Transaction Detected!`;
    } else if (data.prediction === 0) {
      // Add 'safe' class to apply green styling from the CSS
      resultBox.className = "result-box show safe";
      resultBox.innerHTML = `<i class="fa-solid fa-check-circle"></i> Legitimate Transaction`;
    } else {
      // Handle cases where the prediction value is unexpected
      resultBox.className = "result-box show fraud";
      resultBox.innerHTML = `<i class="fa-solid fa-xmark-circle"></i> Error: Unexpected response format.`;
    }
  } catch (err) {
    console.error("Prediction Error:", err);
    // Use the 'fraud' style for connection errors to make them visible
    resultBox.className = "result-box show fraud";
    resultBox.innerHTML = `<i class="fa-solid fa-server"></i> Error: Could not connect to the backend server.`;
  }
});