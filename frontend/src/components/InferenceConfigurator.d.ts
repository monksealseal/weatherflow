import { InferenceConfig } from '../api/types';
interface Props {
    value: InferenceConfig;
    onChange: (config: InferenceConfig) => void;
}
declare function InferenceConfigurator({ value, onChange }: Props): JSX.Element;
export default InferenceConfigurator;
