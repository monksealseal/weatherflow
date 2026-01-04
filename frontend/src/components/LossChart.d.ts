import { MetricEntry, ValidationMetricEntry } from '../api/types';
interface Props {
    train: MetricEntry[];
    validation: ValidationMetricEntry[];
}
declare function LossChart({ train, validation }: Props): JSX.Element;
export default LossChart;
