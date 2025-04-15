import Flow from "@/components/flow";
import "./App.css";
import RecordButton from "./components/record-button";
import UploadButton from "./components/upload-button"
import { SessionProvider } from "./components/session-context";

function App() {
  return (
    <SessionProvider>
      <div className="relative w-screen h-screen padding-0 margin-0">
        <Flow/>
        <div className="relative z-10">
          <RecordButton/>
          <UploadButton/>
        </div>
      </div>
    </SessionProvider>
  );
}

export default App;
