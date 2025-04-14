import Flow from "@/components/flow";
import "./App.css";
import RecordButton from "./components/record-button";
import { useState } from "react";

function App() {
  const [sessionId, setSessionId] = useState("");

  return (
    <div className="relative w-screen h-screen padding-0 margin-0">
      <Flow sessionId={sessionId}/>
      <div className="relative z-10">
        <RecordButton sessionId={sessionId} setSessionId={setSessionId} />
      </div>
    </div>
  );
}

export default App;
