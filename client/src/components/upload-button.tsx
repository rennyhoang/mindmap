import { Input } from '@/components/ui/input'

function UploadButton({sessionId, setSessionId}: {sessionId: string, setSessionId: Function}) {
    const uploadFile = async () => {

    };

    return(
        <>
            <Input type="file"/>
        </>
    );
};

export default UploadButton;