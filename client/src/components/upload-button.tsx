import { useContext } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { SessionContext } from "./session-context";

function UploadButton() {
    const context = useContext(SessionContext);

    if (!context) {
        throw new Error(
          "ChildComponent must be used within a SessionProvider"
        );
    }

    const { sessionId, setSessionId } = context;


    const uploadFile = async () => {
        try {
            const response = await fetch(`http://localhost:8000/uploadfile/${sessionId}`, {
                method: "POST",
            });
            const json = await response.json();
            console.log(json);
        } catch {
            console.log("catch");
        }
    };

    return(
        <form method="post" onSubmit={uploadFile}>
            <Input type="file"/>
            <Button>Submit</Button>
        </form>
    );
};

export default UploadButton;