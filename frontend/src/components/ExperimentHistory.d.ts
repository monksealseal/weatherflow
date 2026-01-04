import { ExperimentRecord } from '../utils/experimentTracker';
import './ExperimentHistory.css';
interface ExperimentHistoryProps {
    onSelectExperiment?: (experiment: ExperimentRecord) => void;
    onCompareExperiments?: (experiments: ExperimentRecord[]) => void;
}
export default function ExperimentHistory({ onSelectExperiment, onCompareExperiments }: ExperimentHistoryProps): JSX.Element;
export {};
