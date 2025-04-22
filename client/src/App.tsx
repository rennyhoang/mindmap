import Flow from "@/components/flow";
import "./App.css";
import RecordButton from "./components/record-button";
import UploadButton from "./components/upload-button";
import Transcript from "./components/transcript";
import { SessionProvider } from "./components/session-context";

function App() {
  return (
    <SessionProvider>
      <div className="relative w-screen h-screen padding-0 margin-0">
        <Flow />
        <div className="w-1/4 absolute flex flex-col gap-4 z-10 m-4 top right-4">
          <UploadButton />
          <RecordButton />
          <Transcript />
        </div>
      </div>
    </SessionProvider>
  );
}

export default App;
