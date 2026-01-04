import { PredictionResult } from '../api/types';
interface Props {
    prediction: PredictionResult;
}
declare function PredictionViewer({ prediction }: Props): JSX.Element;
export default PredictionViewer;
