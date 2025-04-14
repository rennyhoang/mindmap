import Flow from "@/components/flow";
import "./App.css";
import RecordButton from "./components/record-button";
import { useState } from "react";
import { SessionProvider } from "./components/session-context";

function App() {
  const [sessionId, setSessionId] = useState("");

  return (
    <SessionProvider>
      <div className="relative w-screen h-screen padding-0 margin-0">
        <Flow/>
        <div className="relative z-10">
          <RecordButton/>
        </div>
      </div>
    </SessionProvider>
  );
}

export default App;
