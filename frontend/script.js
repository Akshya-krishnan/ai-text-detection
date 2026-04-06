let lastResult = "";
let reasonData = [];


let detected = false;
let reasonShown = false;

async function detect() {
  const text = document.getElementById("text").value;

  if (!text.trim()) {
    alert("Please enter text");
    return;
  }

  try {
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();

    console.log("API Response:", data); 


    document.getElementById("result").innerText =
      data.prediction || "No result";

    
    lastResult = JSON.stringify(data, null, 2);


    reasonData = data.reason || [];

    
    detected = true;
    checkShowDownload();
  } catch (err) {
    console.error(err);
    alert("Server error. Check backend.");
  }
}

function showReason() {
  if (!detected) {
    alert("Please click Detect first!");
    return;
  }

  const list = document.getElementById("reason");
  list.innerHTML = "";

  
  if (!reasonData || reasonData.length === 0) {
    list.innerHTML = "<li>No explanation available</li>";
  } else {
    reasonData.forEach((r) => {
      let li = document.createElement("li");
      li.innerText = r;
      list.appendChild(li);
    });
  }

  
  reasonShown = true;
  checkShowDownload();
}


function checkShowDownload() {
  const downloadBtn = document.getElementById("downloadBtn");
  if (detected && reasonShown) {
    downloadBtn.classList.add("show"); 
  }
}

async function download() {
  if (!lastResult) {
    alert("Nothing to download yet!");
    return;
  }

  try {
    
    const content = `Prediction:\n${document.getElementById("result").innerText}\n\nReason:\n${reasonData.join("\n")}\n\nRaw API Response:\n${lastResult}`;

    const blob = new Blob([content], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "report.txt";
    a.click();
  } catch (err) {
    console.error(err);
    alert("Download failed");
  }
}
