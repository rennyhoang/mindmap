import Flow from "@/components/flow";
import "./App.css";
import RecordButton from "./components/record-button";
import { useState } from "react";

function App() {
  const [sessionId, setSessionId] = useState("");

  return (
    <div className="w-screen h-screen padding-0 margin-0">
      <RecordButton setSessionId={setSessionId}/>
    </div>
  );
}

export default App;
