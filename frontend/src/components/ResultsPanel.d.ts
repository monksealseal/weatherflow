import { ExperimentResult } from '../api/types';
interface Props {
    result: ExperimentResult | null;
    loading: boolean;
    hasConfig: boolean;
}
declare function ResultsPanel({ result, loading, hasConfig }: Props): JSX.Element;
export default ResultsPanel;
